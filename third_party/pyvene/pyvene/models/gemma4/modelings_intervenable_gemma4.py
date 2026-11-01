# Added for InternalModalTransfer; not part of the pyvene v0.1.5 release.
# Distributed under the pyvene Apache-2.0 license (see third_party/pyvene/LICENSE).
"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.
We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""

import torch
from transformers import AutoModelForMultimodalLM

from ..constants import *

gemma4_type_to_module_mapping = {
    "block_input": ("model.language_model.layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("model.language_model.layers[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("model.language_model.layers[%s].mlp.act_fn", CONST_OUTPUT_HOOK),
    "mlp_output": ("model.language_model.layers[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("model.language_model.layers[%s].mlp", CONST_INPUT_HOOK),
    "attention_value_output": ("model.language_model.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": (
        "model.language_model.layers[%s].self_attn.o_proj",
        CONST_INPUT_HOOK,
        (split_head_and_permute, "n_head"),
    ),
    "attention_output": ("model.language_model.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_input": ("model.language_model.layers[%s].self_attn", CONST_INPUT_HOOK),
    "query_output": ("model.language_model.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "key_output": ("model.language_model.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "value_output": ("model.language_model.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "head_query_output": (
        "model.language_model.layers[%s].self_attn.q_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "n_head"),
    ),
    "head_key_output": (
        "model.language_model.layers[%s].self_attn.k_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "n_kv_head"),
    ),
    "head_value_output": (
        "model.language_model.layers[%s].self_attn.v_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "n_kv_head"),
    ),
}

gemma4_type_to_dimension_mapping = {
    "n_head": ("text_config.num_attention_heads",),
    "n_kv_head": ("text_config.num_key_value_heads",),
    "block_input": ("text_config.hidden_size",),
    "block_output": ("text_config.hidden_size",),
    "mlp_activation": ("text_config.intermediate_size",),
    "mlp_output": ("text_config.hidden_size",),
    "mlp_input": ("text_config.hidden_size",),
    "attention_value_output": ("text_config.hidden_size",),
    "head_attention_value_output": (
        "head_dim",
        "text_config.hidden_size/text_config.num_attention_heads",
    ),
    "attention_output": ("text_config.hidden_size",),
    "attention_input": ("text_config.hidden_size",),
    "query_output": ("text_config.hidden_size",),
    "key_output": ("text_config.num_key_value_heads*text_config.hidden_size/text_config.num_attention_heads",),
    "value_output": ("text_config.num_key_value_heads*text_config.hidden_size/text_config.num_attention_heads",),
    "head_query_output": (
        "head_dim",
        "text_config.hidden_size/text_config.num_attention_heads",
    ),
    "head_key_output": (
        "head_dim",
        "text_config.hidden_size/text_config.num_attention_heads",
    ),
    "head_value_output": (
        "head_dim",
        "text_config.hidden_size/text_config.num_attention_heads",
    ),
}

"""gemma4 model with LM head"""
gemma4_lm_type_to_module_mapping = {}
for k, v in gemma4_type_to_module_mapping.items():
    gemma4_lm_type_to_module_mapping[k] = (f"model.{v[0]}",) + v[1:]
gemma4_lm_type_to_dimension_mapping = gemma4_type_to_dimension_mapping

"""gemma4 model with classifier head"""
gemma4_classifier_type_to_module_mapping = {}
for k, v in gemma4_type_to_module_mapping.items():
    gemma4_classifier_type_to_module_mapping[k] = (f"model.{v[0]}",) + v[1:]
gemma4_classifier_type_to_dimension_mapping = gemma4_type_to_dimension_mapping


def create_gemma4(name="google/gemma-4-E2B-it", cache_dir=None, config=None, dtype=torch.bfloat16):
    """Creates a Causal LM model, config, and tokenizer from the given name and revision"""
    from transformers import AutoConfig, AutoModelForMultimodalLM, AutoTokenizer

    if config is None:
        config = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        model = AutoModelForMultimodalLM.from_pretrained(
            name,
            config=config,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        model = AutoModelForMultimodalLM.from_pretrained(config)

    return config, tokenizer, model
