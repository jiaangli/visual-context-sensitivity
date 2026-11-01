# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
import wandb
from accelerate.utils import gather_object
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainerCallback, is_datasets_available
from transformers.trainer import (
    math,
    speed_metrics,
    time,
    tpu_spmd_dataloader,
)
from transformers.trainer_utils import SaveStrategy
from transformers.utils import is_torch_xla_available

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


class MetaQueryTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        self._eval_sample_idx = 0
        self._current_eval_prefix = metric_key_prefix
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    @staticmethod
    def metaquery_eval_data_collator(features):
        batch = {}
        for k in features[0].keys():
            batch[k] = [f[k] for f in features]

        return batch

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.metaquery_eval_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": False,
            "persistent_workers": False,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        return self.accelerator.prepare(eval_dataloader)

    def log_images(self, logs: Dict[str, float]) -> None:
        logs["step"] = self.state.global_step
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        sample_kwargs = {
            "guidance_scale": 0.0,
            "image_guidance_scale": 3.0,
            "num_inference_steps": 30,
            "return_tensor": True,
            "negative_prompt": "",
        }
        captions = inputs.get("captions", [""] * self.args.eval_batch_size)
        with torch.no_grad():
            samples = model.sample_images(**inputs, **sample_kwargs)
        samples = self.accelerator.gather(samples)
        gathered_captions = gather_object(list(captions))
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        prefix = getattr(self, "_current_eval_prefix", "eval")
        start_idx = getattr(self, "_eval_sample_idx", 0)
        image_logs = {
            f"{prefix}/sample_{i + start_idx:03d}": wandb.Image(image, caption=gathered_captions[i])
            for i, image in enumerate(samples)
        }
        self._eval_sample_idx += len(samples)
        self.log_images(image_logs)
        del samples
        gc.collect()

        return (None, None, None)

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        decay = 0.99
        # detect loss spike
        if not hasattr(self, "running_loss"):
            self.running_loss = tr_loss.item()
        self.running_loss = decay * self.running_loss + (1 - decay) * tr_loss.item()

        if tr_loss.item() > 20 * self.running_loss:
            print(f"Loss Spiked: {tr_loss.item()} > 2 * {self.running_loss}")
            self.control.should_training_stop = True

        # detect grad norm spike
        if not hasattr(self, "running_grad_norm"):
            self.running_grad_norm = grad_norm
        self.running_grad_norm = decay * self.running_grad_norm + (1 - decay) * grad_norm

        if grad_norm > 25 * self.running_grad_norm and grad_norm > 1:
            print(f"Grad Norm Spiked: {grad_norm} > 10 * {self.running_grad_norm}")
            self.control.should_training_stop = False
            print("Skipping Step to Avoid Divergence")
            return

        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.cpu().to(torch.float32)
        if np.isnan(grad_norm) or grad_norm > 1e6:
            print(f"NaN grad norm detected in process {self.args.process_index} on {os.uname().nodename}")
            self.control.should_training_stop = True
            print(f"Shut Down Training")

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self.accelerator.gather(tr_loss).mean().item()

            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


class MetaQueryCallback(TrainerCallback):
    @staticmethod
    def print_params(model, modules_to_stat):
        for module in modules_to_stat:
            module_parts = module.split(".")
            curr_module = model
            for part in module_parts:
                curr_module = getattr(curr_module, part)
            num_params = sum(p.numel() for p in curr_module.parameters()) / 1e6
            print(f"{module} num of params: {num_params:.2f}M")

    def on_train_begin(self, args, state, control, model, **kwargs):
        if state.is_world_process_zero:
            stat = []
            for i, (n, p) in enumerate(model.named_parameters()):
                stat.append([i, n, p.shape, p.dtype, p.requires_grad])
            # print(tabulate(stat, headers=["idx", "name", "shape", "dtype", "trainable"]))

            if hasattr(model.model, "connector") and not isinstance(model.model.connector, nn.Identity):
                print(f"connector_in_dim = {model.model.connector_in_dim}")
                print(f"connector_out_dim = {model.model.connector_out_dim}")
                self.print_params(model, ["model.connector"])

            self.print_params(model, ["model.transformer"])
