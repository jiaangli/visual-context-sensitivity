from src.interp.api.model_api import ModelAPI


class Qwen3p5(ModelAPI):
    @staticmethod
    def _layer_type(model, layer_idx):
        return model.config.text_config.layer_types[layer_idx]

    @staticmethod
    def _require_full_attention(model, layer_idx, what):
        layer_type = Qwen3p5._layer_type(model, layer_idx)
        return layer_type
        # if layer_type != "full_attention":
        #     raise ValueError(
        #         f"Layer {layer_idx} is not a full attention layer, but {layer_type}, so {what} is not available."
        #     )

    @staticmethod
    def get_q(model, layer_idx):
        layer_type = Qwen3p5._require_full_attention(model, layer_idx, "q_proj")
        if layer_type == "full_attention":
            return model.model.language_model.layers[layer_idx].self_attn.q_proj
        else:
            return model.model.language_model.layers[layer_idx].linear_attn.in_proj_qkv

    @staticmethod
    def get_k(model, layer_idx):
        layer_type = Qwen3p5._require_full_attention(model, layer_idx, "k_proj")
        if layer_type == "full_attention":
            return model.model.language_model.layers[layer_idx].self_attn.k_proj
        else:
            return model.model.language_model.layers[layer_idx].linear_attn.in_proj_qkv

    @staticmethod
    def get_v(model, layer_idx):
        layer_type = Qwen3p5._require_full_attention(model, layer_idx, "v_proj")
        if layer_type == "full_attention":
            return model.model.language_model.layers[layer_idx].self_attn.v_proj
        else:
            return model.model.language_model.layers[layer_idx].linear_attn.in_proj_qkv

    @staticmethod
    def get_o(model, layer_idx):
        layer_type = Qwen3p5._require_full_attention(model, layer_idx, "o_proj")
        if layer_type == "full_attention":
            return model.model.language_model.layers[layer_idx].self_attn.o_proj
        else:
            return model.model.language_model.layers[layer_idx].linear_attn.out_proj

    @staticmethod
    def get_layer(model, layer_idx):
        return model.model.language_model.layers[layer_idx]

    @staticmethod
    def get_attn(model, layer_idx):
        layer = model.model.language_model.layers[layer_idx]
        return getattr(layer, "self_attn", None) or layer.linear_attention

    @staticmethod
    def get_mlp(model, layer_idx):
        return model.model.language_model.layers[layer_idx].mlp

    @staticmethod
    def N_LAYERS(model):
        return model.model.config.text_config.num_hidden_layers

    @staticmethod
    def N_QO_HEADS(model):
        return model.model.config.text_config.num_attention_heads

    @staticmethod
    def N_KV_HEADS(model):
        return model.model.config.text_config.num_key_value_heads
