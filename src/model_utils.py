import logging
import random

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

logger = logging.getLogger(__name__)


def setup_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # if deterministic:
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     torch.use_deterministic_algorithms(True)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_torch_dtype(device):
    if device == "mps":
        return torch.float16
    elif device == "cuda":
        return torch.bfloat16
    else:
        return torch.float32


def load_model_and_processor(
    model_name_or_path,
    *,
    lora_path=None,
    merge_lora=False,
    quantization_config=None,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    padding_side="left",
    use_cache=True,
    eval_mode=True,
):
    load_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    if lora_path is not None:
        base_model = AutoModelForMultimodalLM.from_pretrained(model_name_or_path, **load_kwargs)
        model = PeftModel.from_pretrained(base_model, lora_path)
        if merge_lora:
            model = model.merge_and_unload()
    else:
        model = AutoModelForMultimodalLM.from_pretrained(model_name_or_path, **load_kwargs)

    model.config.use_cache = use_cache
    # if hasattr(model.config, "text_config"):
    #     model.config.text_config.use_cache = use_cache
    if eval_mode:
        model.eval()

    processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=True)
    processor.tokenizer.padding_side = padding_side

    if hasattr(processor.tokenizer, "chat_template") and processor.tokenizer.chat_template:
        if "enable_thinking" in processor.tokenizer.chat_template:
            processor.tokenizer.chat_template = (
                "{%- set enable_thinking = false %}\n" + processor.tokenizer.chat_template
            )

    return model, processor
