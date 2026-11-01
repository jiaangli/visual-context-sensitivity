import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from nnsight import NNsight
from torch import Tensor
from tqdm import trange

from src.interp.site import MultiSite, Site

from .api.qwen2p5vl import Qwen2p5VL


def get_prob(x: Tensor, indices: Tensor) -> Tensor:
    return x.softmax(-1)[range(len(x)), indices]


def get_probs(logits: Tensor, indices: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    probs = torch.stack(
        [get_prob(logits[i].cpu().detach(), indices.cpu().detach()) for i in range(len(logits))]
    ).float()
    stdv = probs.std(-1).unsqueeze(1)
    mean = probs.mean(-1).unsqueeze(1)
    median = probs.median(-1).values.unsqueeze(1)
    return mean, stdv, median


def get_logits(logits: Tensor, indices: Tensor) -> Tensor:
    batch_range = torch.arange(logits.shape[1]).to(logits.device)
    return logits[:, batch_range, indices.to(logits.device)]


def patch_scope(
    nnmodel: NNsight,
    processor,
    residuals: Tensor,
    verbose: bool = False,
    use_image: bool = False,
    batch_size: Optional[int] = None,
) -> Tensor:
    """
    Modified version with batching support.

    Args:
        nnmodel: The model with nnsight tracing capability.
        processor: The tokenizer/processor.
        residuals: Tensor of shape [num_layers, total_samples, hidden_dim].
        verbose: Whether to show progress bar.
        use_image: Whether to include a dummy image in the prompt.
        batchsize: If provided, process samples in batches of this size.
                   If None or >= total_samples, process all at once.

    Returns:
        Tensor of shape [num_layers, total_samples, vocab_size] containing logits.
    """
    total_samples = residuals.size(1)
    # If batch_size is None or larger than total, fallback to original single-batch logic
    if batch_size is None:
        batch_size = 32
    if batch_size >= total_samples:
        batch_size = total_samples
    num_batches = (total_samples + batch_size - 1) // batch_size

    # Prepare a list to collect logits for each layer across batches
    # shape: [num_layers] each will be a list of tensors to concatenate later
    num_layers = residuals.size(0)
    layer_logits = [[] for _ in range(num_layers)]

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_samples)
        current_batch_size = end - start

        # Slice residuals for current batch
        residuals_batch = residuals[:, start:end, :]  # [num_layers, current_batch_size, hidden_dim]

        id_prompt_target = ["cat -> cat\n1135 -> 1135\nhello -> hello\n?"] * current_batch_size

        id_prompt_data = processor.tokenizer(id_prompt_target, return_tensors="pt", padding=True).to(nnmodel.device)

        # Move to device and remove last two tokens
        id_prompt_data = {
            k: v.to(nnmodel.device)
            if k not in ["pixel_values", "image_grid_thw", "image_position_ids"]
            else v.to(nnmodel.device)
            for k, v in id_prompt_data.items()
        }

        input_token_ids = id_prompt_data["input_ids"]
        del id_prompt_data["input_ids"]

        # Trace each layer for the current batch
        lrange = trange(num_layers, desc=f"Batch {batch_idx + 1}/{num_batches}") if verbose else range(num_layers)
        for i in lrange:
            with nnmodel.trace(input_token_ids, **id_prompt_data, validate=False, scan=False):
                # Inject residual into the last token position of layer i
                nnmodel.model.language_model.layers[i].output[:, -1, :] = residuals_batch[i, :, :]
                logits = nnmodel.lm_head.output[:, -1, :].save()

            layer_logits[i].append(logits.value.detach().cpu())

    # Concatenate batch results for each layer and stack along layer dimension
    all_logits = torch.stack([torch.cat(layer_logits[i], dim=0) for i in range(num_layers)])

    return all_logits


def get_patched_residuals(
    nnmodel: NNsight,
    site,
    source_input_ids: Tensor,
    target_input_ids: Tensor,
    source_attention_mask: Tensor,
    target_attention_mask: Tensor,
    source_pixel_values: Optional[Tensor] = None,
    target_pixel_values: Optional[Tensor] = None,
    source_image_grid_thw: Optional[Tensor] = None,
    target_image_grid_thw: Optional[Tensor] = None,
    source_image_position_ids: Optional[Tensor] = None,
    target_image_position_ids: Optional[Tensor] = None,
    source_mm_token_type_ids: Optional[Tensor] = None,
    target_mm_token_type_ids: Optional[Tensor] = None,
    scan: bool = False,
    validate: bool = False,
    average_site=None,
    return_logits: bool = False,
    return_residuals: bool = True,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    site.reset()
    residuals = [[] for _ in range(len(nnmodel.model.language_model.layers))]
    logits = []

    source_kwargs = {
        "attention_mask": source_attention_mask,
        "scan": scan,
        "validate": validate,
    }
    target_kwargs = {
        "attention_mask": target_attention_mask,
        "scan": scan,
        "validate": validate,
    }

    if source_mm_token_type_ids is not None:
        source_kwargs["mm_token_type_ids"] = source_mm_token_type_ids
    if source_pixel_values is not None:
        source_kwargs["pixel_values"] = source_pixel_values
    if source_image_grid_thw is not None:
        source_kwargs["image_grid_thw"] = source_image_grid_thw
    if source_image_position_ids is not None:
        source_kwargs["image_position_ids"] = source_image_position_ids
    if target_mm_token_type_ids is not None:
        target_kwargs["mm_token_type_ids"] = target_mm_token_type_ids
    if target_pixel_values is not None:
        target_kwargs["pixel_values"] = target_pixel_values
    if target_image_grid_thw is not None:
        target_kwargs["image_grid_thw"] = target_image_grid_thw
    if target_image_position_ids is not None:
        target_kwargs["image_position_ids"] = target_image_position_ids

    with nnmodel.trace(source_input_ids, **source_kwargs):
        site.cache(nnmodel)
    with nnmodel.trace(target_input_ids, **target_kwargs):
        site.patch(nnmodel)
        if average_site is not None:
            average_site.patch(nnmodel)
        if return_residuals:
            for i in range(len(nnmodel.model.language_model.layers)):
                residuals[i].append(nnmodel.model.language_model.layers[i].output[:, -1, :].save())

        if return_logits:
            logits.append(nnmodel.lm_head.output[:, -1, :].save())
    if return_residuals:
        for i in range(len(nnmodel.model.language_model.layers)):
            residuals[i][-1] = residuals[i][-1].value.detach().cpu()

    residuals_out = (
        torch.stack([torch.cat([r.detach() for r in res]) for res in residuals]) if return_residuals else None
    )
    logits_out = torch.stack([logit.detach() for logit in logits]) if return_logits else None
    torch.cuda.empty_cache()

    return residuals_out, logits_out


def get_num_layers(model) -> int:
    if hasattr(model, "config") and hasattr(model.config, "text_config"):
        return getattr(model.config.text_config, "num_hidden_layers", 36)
    elif hasattr(model, "model") and hasattr(model.model.language_model, "layers"):
        return len(model.model.language_model.layers)
    else:
        raise ValueError("Cannot determine number of layers")


def config_to_site(
    config: Dict[str, Any],
    api=Qwen2p5VL,
    model=None,
) -> MultiSite:
    """
    Convert a site config dict to a MultiSite object.

    This follows the reference implementation pattern where config is like:
        {"o": {"layers": [0, 1, 2], "heads": [0, 1, 2, ...]}}

    For VLMs, we adapt this to work with the model's architecture.

    Args:
        config: Dict with site names as keys and dicts containing "layers" and
                optionally "heads" as values.
        api: The ModelAPI to use (e.g. Qwen2p5VL).
        model: The model to get head count from. If None, uses default of 32.

    Returns:
        MultiSite object containing all specified sites.
    """
    ms = []

    # We might need model for num heads, or if it's missing, default
    try:
        n_heads = api.N_QO_HEADS(model) if model is not None else 32
    except Exception:
        n_heads = 32

    for head_name in config:
        head_config = config[head_name]
        layers = head_config.get("layers", [])
        heads = head_config.get("heads", list(range(n_heads)))

        for layer in layers:
            for head in heads:
                site = Site.get_site(
                    api,
                    head_name,
                    layer,
                    head=head,
                    seq_pos=torch.tensor([-1]),
                    cache_name="site1",
                )
                ms.append(site)

    return MultiSite(ms)
