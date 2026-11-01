import torch
from torch import Tensor, nn


def _as_hidden_states(x):
    return x[0] if isinstance(x, tuple) else x


class SteerHook:
    def __init__(
        self,
        proj,
        layer,
        value=None,
        device=None,
        last_token_only=True,
        prefill_only=False,
    ):
        if prefill_only and not last_token_only:
            raise ValueError("prefill_only can only be used with last_token_only=True")
        self.proj = proj
        self.value = value
        self.layer = layer
        self.hook = None
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_token_only = last_token_only
        self.proj.to(self.device)
        self._activated = True
        self.prefill_only = prefill_only
        self._position_mask: Tensor | None = None

    def set_value(self, value):
        print("setting value", value)
        self.value = value
        self.activate()

    def set_position_mask(self, mask: Tensor):
        self._position_mask = mask

    def get_value(self, output):
        return self.value.to(self.device)

    def __call__(self, module, input, output):
        if not self._activated:
            return output
        output = _as_hidden_states(output)
        input_hidden_states = _as_hidden_states(input)

        if self.prefill_only and self._position_mask is None and output.shape[1] == 1:
            return output

        assert self.hook is not None, "Hook is not set"
        source_value = self.get_value(input_hidden_states)
        assert source_value is not None, "Value is not set"
        assert source_value.shape[0] == output.shape[0], (
            f"Value shape {source_value.shape} does not match input shape {input_hidden_states.shape}"
        )
        if self._position_mask is not None:
            mask = self._position_mask.to(self.device, dtype=torch.bool)
            if mask.shape != output.shape[:2]:
                raise ValueError(f"Positional mask shape {mask.shape} does not match output shape {output.shape[:2]}")
            if not torch.any(mask):
                return output
            target = output[mask]
            if source_value.ndim != 1:
                raise ValueError(f"Source value must be 1D when using positional mask, got shape {source_value.shape}")
            source_value = source_value[:, None].expand_as(mask)[mask]
        elif self.last_token_only:
            target = output[:, -1, :]
        else:
            target = output

        # source_value = self.get_value(input_hidden_states)
        # h_t = (I-P) h_t + P h_s
        intervened_target = self.proj.constant_forward(source_value, target)

        if self._position_mask is not None:
            output[mask] = intervened_target
        elif self.last_token_only:
            output[:, -1, :] = intervened_target
        else:
            output[:, :, :] = intervened_target
        return output

    def remove(self):
        assert self.hook is not None, "Hook is not set"
        self.hook.remove()
        self.hook = None

    def attach(self, model):
        assert self.hook is None, "Hook is already set"
        base_model = model
        while hasattr(base_model, "model"):
            base_model = base_model.model
        if not hasattr(base_model, "layers"):
            base_model = base_model.language_model
        self.hook = base_model.layers[self.layer].register_forward_hook(self)
        return self.hook

    def deactivate(self):
        self._activated = False

    def activate(self):
        self._activated = True


class CtxPriorHook(SteerHook):
    def __init__(
        self,
        proj: nn.Module,
        layer: int,
        prior_value: float = 6.0,
        context_value: float = -6.0,
        last_token_only: bool = True,
        prefill_only: bool = False,
    ):
        super().__init__(proj, layer, last_token_only=last_token_only, prefill_only=prefill_only)
        self.prior_value = prior_value
        self.context_value = context_value
        self.value = self.prior_value

    def get_value(self, output: tuple) -> Tensor:
        output = _as_hidden_states(output)
        if isinstance(self.value, (int, float)):
            batch_size = output.shape[0]
            return torch.tensor(self.value, dtype=torch.float32).repeat(batch_size).to(self.device)
        return self.value.to(self.device)

    def set_constant_prior(self) -> None:
        self.value = self.prior_value
        self.activate()

    def set_constant_context(self) -> None:
        self.value = self.context_value
        self.activate()

    def set_context_prior(self, value: Tensor) -> None:
        value = value.to(torch.float32)
        self.value = self.context_value * value + self.prior_value * (1 - value)
        self.activate()


class BinaryHook(CtxPriorHook):
    def __init__(
        self,
        proj: nn.Module,
        layer: int,
        value_a: float = 6.0,
        value_b: float = -6.0,
        last_token_only: bool = True,
        prefill_only: bool = False,
    ):
        super().__init__(
            proj,
            layer,
            prior_value=value_a,
            context_value=value_b,
            last_token_only=last_token_only,
            prefill_only=prefill_only,
        )
        self._binary_mask: Tensor | None = None

    def set_binary(self, mask: Tensor) -> None:
        self._binary_mask = mask.to(torch.float32)
        self.activate()

    def get_value(self, output) -> Tensor:
        if self._binary_mask is not None:
            batch_size = _as_hidden_states(output).shape[0]
            mask = self._binary_mask[:batch_size].to(self.device)
            return self.context_value * mask + self.prior_value * (1 - mask)
        return super().get_value(output)


class FeatureCollectionHook:
    def __init__(
        self,
        proj: nn.Module,
        layer: int,
        device=None,
    ):
        self.proj = proj
        self.layer = layer
        self.hook = None
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proj.to(self.device)
        self.features: list = []
        self._activated = True

    def __call__(self, module: nn.Module, input: tuple, output: tuple) -> tuple:
        if not self._activated:
            return output

        base = _as_hidden_states(output)[:, -1, :].detach()
        coordinates = self.proj.project(base)
        self.features.append(coordinates.cpu())
        return output

    def remove(self) -> None:
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def attach(self, model: nn.Module) -> None:
        assert self.hook is None, "Hook is already attached"
        self.hook = self._get_layer_module(model).register_forward_hook(self)

    def _get_layer_module(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[self.layer]
        elif hasattr(model, "language_model"):
            return model.language_model.model.layers[self.layer]
        else:
            raise ValueError(f"Cannot find layer {self.layer} in model architecture")

    def reset(self) -> None:
        self.features = []

    def get_features(self) -> Tensor:
        if not self.features:
            raise ValueError("No features collected")
        return torch.cat(self.features, dim=0)

    def deactivate(self) -> None:
        self._activated = False

    def activate(self) -> None:
        self._activated = True
