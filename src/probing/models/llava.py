import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers.image_processing_utils import select_best_resolution

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class DataPrep(Dataset):
    def __init__(
        self,
        raw_datasets,
        processor,
        conversation_template=None,
        max_new_tokens=300,
        B=2,
        input_setting="image",
        caption_img_ids=None,
        captions=None,
        dataset_name=None,
        experiment_type=None,
    ):
        self.raw_datasets = raw_datasets
        self.processor = processor
        self.conversation_template = conversation_template
        self.max_new_tokens = max_new_tokens
        self.B = B
        self.input_setting = input_setting
        self.caption_img_ids = caption_img_ids
        self.captions = captions
        self.dataset_name = dataset_name
        self.experiment_type = experiment_type

    def __len__(self):
        return math.ceil(len(self.raw_datasets) / self.B)

    def __getitem__(self, idx):
        # Since it has to be a batch, we need to get the current index, which will be idx * B
        curr_idx = idx * self.B
        curr_bs = min(self.B, len(self.raw_datasets) - curr_idx)

        if self.experiment_type != "representation":
            images = self.raw_datasets[curr_idx : curr_idx + curr_bs]["image"]
            prompts = [
                self.processor.apply_chat_template(self.conversation_template, add_generation_prompt=True)
                for _ in range(curr_bs)
            ]

            # Prepare the input data for the model
            inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt")
        else:
            assert self.captions is not None and self.caption_img_ids is not None and self.dataset_name is not None
            messages = text_prompts(
                curr_bs,
                self.raw_datasets[curr_idx : curr_idx + curr_bs],
                self.captions,
                self.caption_img_ids,
                self.input_setting,
                self.dataset_name,
            )
            prompts = [
                self.processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in messages
            ]
            inputs = self.processor(
                images=self.raw_datasets[curr_idx : curr_idx + curr_bs]["image"]
                if "image" in self.input_setting
                else None,
                text=prompts,
                padding=True,
                return_tensors="pt",
            )
        # Ensure data is on the correct device
        return inputs


def generate_captions(
    model_name_or_path: str,
    model,
    processor,
    raw_datasets,
    B: int,
    conversation_template: List[Dict],
    MAX_NEW_TOKENS: int,
    saving_path: Path,
    input_setting: str = "caption",
    experiment_type=None,  # captioning or open_qa
) -> None:
    """
    Generate captions for the given dataset using the specified model.
    """
    model.eval()
    saving = defaultdict(list)

    prepared_data = DataPrep(
        raw_datasets,
        processor,
        conversation_template,
        MAX_NEW_TOKENS,
        B,
        input_setting,
        experiment_type,
    )
    dataloader = DataLoader(prepared_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    num_steps = len(dataloader)

    for step, inputs in tqdm.tqdm(enumerate(dataloader), total=num_steps):
        inputs = {input_k: input_v.squeeze(0).to(model.device) for input_k, input_v in inputs.items()}

        with torch.inference_mode():
            with torch.autocast(dtype=torch.bfloat16, device_type=device):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=processor.tokenizer.eos_token_id,
                ).cpu()  # type: ignore
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        # if "Category" in raw_datasets.column_names:
        #     saving["image_Category"].extend(raw_datasets[i : i + batch_size]["Category"])
        saving["caption"].extend(output_text)
    saving["image_id"] = raw_datasets["ID"]
    saving["image_label"] = raw_datasets["label"]

    save_captions(saving, saving_path, model_name_or_path)


def get_representations(
    model,
    processor,
    raw_datasets,
    B: int,
    db_con,
    dense_caption_path: str,
    input_setting: str,
    dataset_name: str,
    experiment_type: str = "representation",
) -> None:
    """
    Get representations from the model and save them to the database.
    """
    img_ids, dense_caption, zero_shot_answer = load_captions(dense_caption_path)
    prepared_data = DataPrep(
        raw_datasets,
        processor,
        B=B,
        input_setting=input_setting,
        caption_img_ids=img_ids,
        captions=dense_caption,
        dataset_name=dataset_name,
        experiment_type=experiment_type,
    )
    dataloader = DataLoader(prepared_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    num_steps = len(dataloader)
    log_step = 64

    hooks = []
    saving = defaultdict(list)

    model.eval()
    logger.info(model)

    for step, inputs in tqdm.tqdm(enumerate(dataloader), total=num_steps):
        inputs = {input_k: input_v.squeeze(0).to(model.device) for input_k, input_v in inputs.items()}
        if "image" == input_setting:
            model = hook_model(model, inputs, input_setting, saving, hooks)
        with torch.inference_mode():
            with torch.autocast(dtype=torch.bfloat16, device_type=device):
                outputs = model(**inputs, return_dict=True, output_hidden_states=True)
            outputs = torch.stack(outputs.hidden_states, dim=0)  # [num_layers, batch_size, seq_len, hidden_dim]
            # Get last token representations for all layers and batches at once
            # [num_layers, batch_size, seq_len, hidden_dim] -> [num_layers, batch_size, hidden_dim]
            last_tokens = outputs[:, :, -1, :].to(torch.float32).cpu().detach().numpy()
            # Store results for each layer
            for layer_idx, layer_output in enumerate(last_tokens):
                saving[f"llm_{layer_idx}_last"].extend(layer_output)

        if step % log_step == 0 or step == num_steps - 1:
            df = pd.DataFrame(saving)
            # breakpoint()
            if step == 0:
                db_con.execute("CREATE TABLE IF NOT EXISTS data AS SELECT * FROM df")
            else:
                db_con.execute("INSERT INTO data SELECT * FROM df")
            for key in saving:
                saving[key] = []
            del df


def text_prompts(
    batch_size: int,
    batch: pd.DataFrame,
    dense_caption: List[str],
    img_ids: List[str],
    input_setting: str,
    dataset_name: str,
) -> List:
    """Generate text prompts based on the input settings."""

    question = ZERO_SHOT_PROMPTS[dataset_name]
    if "image" not in input_setting:
        question = question.replace("image", "description")
    settings_config = {
        "image": {"content": [{"type": "image"}], "instruction": "", "caption_required": False},
        "caption": {"content": [], "instruction": "", "caption_required": True},
        "caption+question": {
            "content": [],
            "instruction": f"Use the above caption to answer the following question.\n{question}",
            "caption_required": True,
        },
        "image+question": {
            "content": [{"type": "image"}],
            "instruction": f"{question[:-1]} in the image?",
            "caption_required": False,
        },
        "image+caption+question": {
            "content": [{"type": "image"}],
            "instruction": f"Use the above caption and the image to answer the following question.\n{question}",
            "caption_required": True,
        },
    }

    if input_setting not in settings_config:
        raise ValueError("Invalid input settings")

    config = settings_config[input_setting]
    messages = []

    for i in range(batch_size):
        content = config["content"].copy()
        sample_id = batch.get("ID", [None])[i] if config["caption_required"] else None

        if sample_id and config["caption_required"]:
            caption_text = dense_caption[img_ids.index(sample_id)] + "\n"
            if caption_text or config["instruction"]:
                content.append({"type": "text", "text": f"{caption_text}{config['instruction']}".strip()})
        elif config["instruction"]:
            content.append({"type": "text", "text": config["instruction"]})

        messages.append([{"role": "user", "content": content}])

    return messages


# Copied from transformers.models.llava_next.modeling_llava_next.image_size_to_num_patches
def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """
    Calculate the number of patches after the preprocessing for images of any resolution.

    Args:
        image_size (`torch.LongTensor` or `np.ndarray` or `Tuple[int, int]`):
            The size of the input image in the format (height, width). ?
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        int: the number of patches
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)  # type: ignore
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches


def hook_model(model, inputs, input_setting: str, saving: Dict, hook_handles: list):
    """
    Hook the model to save intermediate outputs.
    one image can be [7, 729, 1152], use the default strategy [7,729, 1152]
    LLAVA model use a base feature concatenate crops features as a whole image feature.
    [7,729,1152] -> [1,729,1152], [6, 729, 1152]

    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L459
    """
    # print(inputs)
    # breakpoint()
    image_num_patches = [
        image_size_to_num_patches(
            image_size=imsize,
            grid_pinpoints=model.config.image_grid_pinpoints,
            patch_size=model.config.vision_config.image_size,
        )
        for imsize in inputs["image_sizes"]
    ]

    for handle in hook_handles:
        handle.remove()
    hook_handles.clear()  # Reset the list

    def hook_fn(name):
        def hook(module, module_input, module_output):
            module_output = module_output[0] if isinstance(module_output, tuple) else module_output
            # Since the model will select best resolution for the image, which will add padding to the image then get the number of patches
            # The number of patches may have padding value in it, so we need to do the post processing to get the real values like the model did.
            # image size: [1920,1440] -> [1, 21, 3, 384, 384] -> [21, 729, 1152] -> base: [1, 729, 1152], crops: [20, 729, 1152] -> [20, 729, 1152]
            # After get anyres_image_grid_shape, [5, 4, 27, 27, 1152] -> [135, 108, 1152] -> after unpad_image -> [135, 102, 1152]
            if "vision" in name or "connector" in name:
                v_embs = torch.split(module_output, image_num_patches, dim=0)
                for idx, output in enumerate(v_embs):
                    if output.shape[0] > 1:
                        base_image_feature = output[0]
                        image_feature = output[1:]
                        height = width = model.config.vision_config.image_size // model.config.vision_config.patch_size
                        if height * width != base_image_feature.shape[0]:
                            raise ValueError("The number of patches is not consistent with the image size.")
                        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                            inputs["image_sizes"][idx],
                            model.config.image_grid_pinpoints,
                            model.config.vision_config.image_size,
                        )
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, inputs["image_sizes"][idx])
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    saving[name].append(image_feature.cpu().detach().mean(dim=0).to(torch.float32).numpy())
            # else:
            #     # # saving[name].append(output.cpu().detach())
            #     breakpoint()
            #     c_embs = torch.split(module_output, image_num_patches, dim=0)
            #     for idx, output in enumerate(c_embs):
            #         saving[name].append(output.mean(dim=0).cpu().detach().to(torch.float32).numpy())
            #     # saving[name].append(
            #     #     output[:, 1:, :].mean(dim=0).mean(dim=0).cpu().detach().to(torch.float32).numpy()
            #     # )
            #     # saving[name + "_base"].append(
            #     #     output[0, 1:, :].mean(dim=0).cpu().detach().to(torch.float32).numpy()
            #     # )
            #     # breakpoint()
            #     # torch.Size([7, 729, 1152])

        return hook

    # Register hooks on each layer of CLIP encoder in vision_tower
    for idx, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
        v_handle = layer.register_forward_hook(hook_fn(name=f"vision_{idx}"))
        hook_handles.append(v_handle)
    c_handle = model.multi_modal_projector.register_forward_hook(hook_fn(name="connector"))
    hook_handles.append(c_handle)

    return model


# Copied from transformers.models.llava_next.modeling_llava_next.get_anyres_image_grid_shape
def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)  # type: ignore
    return height // patch_size, width // patch_size


# Copied from transformers.models.llava_next.modeling_llava_next.unpad_image
def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    if not isinstance(original_size, (list, tuple)):
        if not isinstance(original_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(original_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        original_size = original_size.tolist()
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(round(original_height * scale_factor, 7))
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(round(original_width * scale_factor, 7))
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor
