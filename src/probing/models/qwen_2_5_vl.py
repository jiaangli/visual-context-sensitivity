import logging
import math
from collections import defaultdict
from typing import Dict

import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class DataPrep(Dataset):
    def __init__(
        self,
        raw_datasets,
        processor,
        max_new_tokens=300,
        B=2,
        dataset_name=None,
    ):
        self.raw_datasets = raw_datasets
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.B = B
        self.dataset_name = dataset_name

    def __len__(self):
        return math.ceil(len(self.raw_datasets) / self.B)

    def __getitem__(self, idx):
        # Since it has to be a batch, we need to get the current index, which will be idx * B
        curr_idx = idx * self.B
        curr_bs = min(self.B, len(self.raw_datasets) - curr_idx, len(self.raw_datasets))
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_i},
                    ],
                },
            ]
            for prompt_i in self.raw_datasets[curr_idx : curr_idx + curr_bs]["prompt"]
        ]
        prompts = [
            self.processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in messages
        ]
        inputs = self.processor(
            images=[img for img in self.raw_datasets[curr_idx : curr_idx + curr_bs]["image"]],
            text=prompts,
            padding=True,
            return_tensors="pt",
        )
        # Ensure data is on the correct device
        return inputs


def get_representations(
    model,
    processor,
    raw_datasets,
    B: int,
    db_con,
    dataset_name: str,
    num_workers: int = 0,
    log_step: int = 64,
) -> None:
    """
    Get representations from the model and save them to the database.
    """
    prepared_data = DataPrep(
        raw_datasets,
        processor,
        B=B,
        dataset_name=dataset_name,
    )
    # batch size should be 1 in dataloader, since we have already batched in the dataset
    # Note: DON'T USE SHUFFLE HERE, otherwise the LABELS will be messed up
    dataloader = DataLoader(prepared_data, shuffle=False, num_workers=num_workers, pin_memory=True)
    num_steps = len(dataloader)

    # saving = {"image_id": [], "image_label": [], "label_name": []}
    saving = defaultdict(list)

    model.eval()
    logger.info(model)

    for step, inputs in tqdm.tqdm(enumerate(dataloader), total=num_steps):
        inputs = {input_k: input_v.squeeze(0).to(model.device) for input_k, input_v in inputs.items()}
        # image_grid_thw will have the number of patches in the each image, which will be used to split the output
        # e.g. for two images (bs=2), pixel values is torch.Size([3768, 1176]) and 'image_grid_thw': tensor([[ 1, 42, 64], [ 1, 30, 36]])
        # then 42 * 64 = 2688 and 30 * 36 = 1080, so the pixel values will be split into two parts of size 2688 and 1080 for two images
        hooks = hook_model(model, inputs, saving)
        with torch.inference_mode():
            with torch.autocast(dtype=torch.bfloat16, device_type=device):
                outputs = model(**inputs, return_dict=True, output_hidden_states=True)
            outputs = torch.stack(outputs.hidden_states, dim=0)  # [num_layers, batch_size, seq_len, hidden_dim]
            # Get last token representations for all layers and batches at once
            # [num_layers, batch_size, hidden_dim]
            last_tokens = outputs[:, :, -1, :].to(torch.float32).cpu().detach().numpy()
            # Store results for each layer
            for layer_idx, layer_output in enumerate(last_tokens):
                saving[f"llm_{layer_idx}_last"].extend(layer_output)

        if step % log_step == 0 or step == num_steps - 1:
            df = pd.DataFrame(saving)
            if step == 0:
                db_con.execute("CREATE TABLE IF NOT EXISTS data AS SELECT * FROM df")
            else:
                db_con.execute("INSERT INTO data SELECT * FROM df")
            for key in saving:
                saving[key] = []
            del df

        for handle in hooks:
            handle.remove()
        del inputs, hooks, outputs, last_tokens
    # Save any remaining data
    del prepared_data, saving, dataloader


def hook_model(model, inputs, saving: Dict):
    """
    Hook the model to save intermediate outputs.
    """
    img_grid_idx = (inputs["image_grid_thw"][:, 1] * inputs["image_grid_thw"][:, 2]).tolist()
    input_ids = inputs["input_ids"]
    img_mask = (input_ids == model.config.image_token_id).unsqueeze(-1)

    hook_handles = []

    def hook_fn(name: str):
        def hook(module, module_input, module_output):
            module_output = module_output[0] if isinstance(module_output, tuple) else module_output
            if "vision" in name:
                # for batch size > 1, split the output based on the image grid size
                v_embs = torch.split(module_output, img_grid_idx, dim=0)
                for idx, output in enumerate(v_embs):
                    saving[name].append(output.mean(dim=0).cpu().detach().to(torch.float32).numpy())
            elif "connector" in name:
                img_mask.to(module_output.dtype)
                c_embs = torch.zeros(input_ids.size(0), input_ids.size(1), model.config.hidden_size).to(
                    img_mask.device, module_output.dtype
                )
                c_mask = img_mask.expand_as(c_embs)
                c_embs = c_embs.masked_scatter(c_mask, module_output)
                for idx, output in enumerate(c_embs):
                    output = output[~(output == 0).all(dim=1)]
                    saving[name].append(output.mean(dim=0).cpu().detach().to(torch.float32).numpy())

        return hook

    for idx, layer in enumerate(model.visual.blocks):
        v_handle = layer.register_forward_hook(hook_fn(name=f"vision_{idx}"))
        hook_handles.append(v_handle)
    # merger will extend the dim based on spatial grid size to reduce the image tokens
    # e.g. [2, 1000, 1280] -> [2, 1000 / spatial_grid_size**2, 1280 * spatial_grid_size**2]
    c_handle = model.visual.merger.register_forward_hook(hook_fn(name="connector"))
    hook_handles.append(c_handle)

    return hook_handles
