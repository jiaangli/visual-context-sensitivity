# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import PIL.Image
import safetensors
import torch
import transformers
import wandb
import yaml
from accelerate.utils import release_memory
from dotenv import load_dotenv
from PIL import PngImagePlugin
from transformers.trainer_utils import get_last_checkpoint

from src.eval_results import (
    DEFAULT_EVAL_ROOT,
    configure_run_logging,
    prepare_eval_run,
    write_json_result,
)
from src.imagegen.metaquery.dataset import get_train_datasets
from src.imagegen.metaquery.metaquery import MetaQuery, MetaQueryConfig
from src.imagegen.metaquery.trainer import MetaQueryCallback, MetaQueryTrainer
from src.imagegen.metaquery.trainer_utils import (
    find_newest_checkpoint,
    get_full_dirs,
    possible_override_args,
)
from src.model_utils import setup_seed

load_dotenv()

# datasets.disable_caching()
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_PROJECT"] = "MetaQuery"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

PIL.Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class OverrideArguments:
    config_file: str = None


@dataclass
class ModelArguments:
    _gradient_checkpointing: bool = True
    vae_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    in_channels: int = 32
    vae_downsample_f: int = 32
    noise_scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    mllm_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    diffusion_model_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    loss_type: str = "flow"
    num_metaqueries: int = 64
    modules_to_freeze: tuple[str] = ()
    modules_to_unfreeze: tuple[str] = ()
    max_input_text_tokens: int = 256
    connector_num_hidden_layers: int = 24
    llm_layer_idx: int = -1
    # system_prompt: str = "You will be given an image or its caption. Please describe the content of the image in detail in your own words."
    system_prompt: str = ""


@dataclass
class DataArguments:
    train_datasets: dict[str, float] = field(
        default_factory=lambda: {
            "default_dataset": -1,
        }
    )
    eval_dataset: str = "cc12m_t2i"
    target_image_size: int = 512
    # Blank-image augmentation for raw-image ImageNet shards; defaults match
    # app/imagegen/preprocess.py, which bakes the same augmentation into latent shards.
    blank_ratio: float = 0.1
    blank_min_image_size: int = 224
    blank_max_aspect_ratio: float = 2.0


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    base_dir: str = "/path/to/base_dir"
    output_dir: str = "output"
    data_dir: str = "./dataset/hf_data"
    tensorboard_logging_dir: str = "logs"
    eval_on_start: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    eval_delay: int = 0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optim: str = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr": 1e-5})
    warmup_steps: int = 5000
    # warmup_ratio: float = 0.01
    logging_steps: int = 1
    save_steps: int = 1000
    save_total_limit: int = 1
    restore_callback_states_from_checkpoint: bool = True
    seed: int = 42
    data_seed: int = 42
    bf16: bool = True
    tf32: bool = True
    dataloader_num_workers: int = 8
    datasets_num_proc: int = os.getenv("OMP_NUM_THREADS", 12)
    dataloader_persistent_workers: bool = False
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    remove_unused_columns: bool = False
    run_name: str = "test"
    report_to: str = "wandb"
    ddp_find_unused_parameters: bool = False
    overwrite_output_dir: bool = False
    resume_from_checkpoint: str = None
    expid: str = "metaquery_train"
    eval_category: str = "metaquery"
    output_root: str = DEFAULT_EVAL_ROOT
    timestamp: str = None
    output_name: str = "metaquery_train.json"

    def __post_init__(self):
        if self.accelerator_config is None:
            self.accelerator_config = {"dispatch_batches": False, "split_batches": False}
        try:
            self = possible_override_args(override_args, self)
            self = get_full_dirs(self)
        except (FileNotFoundError, yaml.YAMLError) as exc:
            print(f"Failed to load override config: {exc}")
        super().__post_init__()
        # accelerator_config = field(default_factory=lambda: {"dispatch_batches": False, "split_batches": False})


if __name__ == "__main__":
    override_parser = transformers.HfArgumentParser((OverrideArguments))
    override_args = override_parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    parser = transformers.HfArgumentParser((OverrideArguments, ModelArguments, DataArguments, TrainingArguments))
    _, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args = possible_override_args(override_args, model_args, data_args)
    eval_run = prepare_eval_run(
        category=training_args.eval_category,
        expid=training_args.expid,
        output_root=training_args.output_root,
        timestamp=training_args.timestamp,
        output_name=training_args.output_name,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        configure_run_logging(eval_run, kind="metaquery-training")
    os.environ.setdefault("WANDB_DIR", str(eval_run.artifact_dir("wandb")))
    logger.info(f"LLM layer: {model_args.llm_layer_idx}, max input text tokens: {model_args.max_input_text_tokens}")
    assert data_args.target_image_size % model_args.vae_downsample_f == 0, (
        f"Image size must be divisible by {model_args.vae_downsample_f}"
    )
    input_size = data_args.target_image_size // model_args.vae_downsample_f

    # Seed before the MetaQuery modules are randomly initialized; the Trainer only seeds later.
    setup_seed(training_args.seed, deterministic=training_args.full_determinism)

    if training_args.resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint = find_newest_checkpoint(training_args.resume_from_checkpoint)
        with torch.device("cpu"):
            model = MetaQuery(config=MetaQueryConfig(input_size=input_size, **model_args.__dict__))
            missing, unexpected = model.load_state_dict(
                safetensors.torch.load_file(os.path.join(training_args.resume_from_checkpoint, "model.safetensors")),
                strict=False,
            )
            print(missing)
            print(unexpected)
            print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
    else:
        model = MetaQuery(config=MetaQueryConfig(input_size=input_size, **model_args.__dict__))

    model = model.to(torch.bfloat16)
    # model.compile(dynamic=False)
    # model = torch.compile(model)
    # training_args.resume_from_checkpoint = None

    with training_args.main_process_first(local=False):
        train_dataset, eval_dataset, gt_images, collate_fn = get_train_datasets(
            data_args,
            training_args,
            model_args,
            model.get_tokenize_fn(),
            model.get_tokenizer(),
        )

    trainer = MetaQueryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[MetaQueryCallback()],
    )
    trainer.log_images({"gt_images": [wandb.Image(image) for image in gt_images]})

    training_args.output_dir = str(os.path.join(training_args.output_dir, training_args.run_name))
    if trainer.is_world_process_zero():
        if training_args.overwrite_output_dir and os.path.exists(training_args.output_dir):
            shutil.rmtree(training_args.output_dir)
        # print(f"Training dataset size: {len(train_dataset)}")

    while trainer.state.global_step == 0 or trainer.state.global_step < training_args.max_steps:
        if trainer.state.global_step > 0:
            trainer.control.should_training_stop = False
            trainer.args.eval_on_start = False
            trainer.model = model
            (trainer.model_wrapped,) = release_memory(trainer.model_wrapped)
            trainer.model_wrapped = trainer.model
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)

    if trainer.is_world_process_zero():
        write_json_result(
            eval_run,
            {
                "checkpoint_path": training_args.output_dir,
                "global_step": trainer.state.global_step,
                "best_metric": trainer.state.best_metric,
                "config": {
                    "model": vars(model_args),
                    "data": vars(data_args),
                    "training": vars(training_args),
                },
            },
            kind="metaquery-training",
            metadata={"run_name": training_args.run_name},
        )
