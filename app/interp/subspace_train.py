import hashlib
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from transformers import (
    HfArgumentParser,
    get_linear_schedule_with_warmup,
)

from src.eval_results import (
    DEFAULT_EVAL_ROOT,
    configure_run_logging,
    prepare_eval_run,
    write_json_result,
)
from src.interp.das import LowRankOrthogonalProjection, compute_loss, compute_metrics, inputs_collator
from src.interp.data import (
    DEFAULT_DAS_TRAIN_PAIRS,
    build_das_datasets,
    build_full_answer_das_dataset,
    filter_by_correct_predictions,
    filter_by_full_answer_preferences,
    full_answer_das_collator,
    prepare_vlm_patching_data,
)
from src.interp.decoding import get_num_layers
from src.model_utils import get_device, get_torch_dtype, load_model_and_processor, setup_seed
from src.sft.preference_eval import candidate_token_ids
from third_party.pyvene.pyvene import (
    DistributedRepresentationIntervention,
    IntervenableConfig,
    IntervenableModel,
    TrainableIntervention,
)

load_dotenv()
torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # filename="train_das.log",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_STEERING_TRAIN = Path("dataset/whatifvis/spatial_temporal_train.jsonl")
DEFAULT_STEERING_VAL = Path("dataset/whatifvis/spatial_temporal_val.jsonl")


@dataclass
class TrainDASArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct", metadata={"help": "Base model path"})
    lora_path: Optional[str] = field(default=None, metadata={"help": "Path to LoRA adapter (if any)"})
    train_data: str = field(
        default=str(DEFAULT_STEERING_TRAIN),
        metadata={"help": "Path to VLM dataset jsonl"},
    )
    val_data: str = field(default=str(DEFAULT_STEERING_VAL), metadata={"help": "Validation split ratio"})
    image_root: Optional[str] = field(default="./dataset/whatifvis", metadata={"help": "Root directory for images"})
    use_image: bool = field(
        default=True,
        metadata={"help": "Disable image inputs (text-only mode)"},
    )
    max_samples: Optional[int] = field(
        default=DEFAULT_DAS_TRAIN_PAIRS, metadata={"help": "Maximum number of samples to load"}
    )
    filter_batch_size: int = field(default=16, metadata={"help": "Batch size for correct-sample filtering"})
    filter_only: bool = field(
        default=False,
        metadata={"help": "Only filter correct samples and save to cache, skip DAS training"},
    )
    skip_layer_search: bool = field(
        default=False,
        metadata={"help": "Skip auto-search and use target_layer directly"},
    )

    das_rank: int = field(default=1, metadata={"help": "Low-rank DAS dimension"})
    das_epochs: int = field(default=1, metadata={"help": "DAS training epochs"})
    das_lr: float = field(default=1e-3, metadata={"help": "DAS learning rate"})
    train_batch_size: int = field(default=8, metadata={"help": "Train batch size"})
    eval_batch_size: int = field(default=8, metadata={"help": "Eval batch size"})
    grad_acc_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps"})
    warmup_ratio: float = field(default=0.01, metadata={"help": "Warmup ratio for LR scheduler"})
    objective: str = field(default="full_answer", metadata={"help": "DAS objective: 'full_answer' or 'first_token'"})
    filter_criterion: str = field(
        default="match_objective",
        metadata={"help": "Filtering criterion: 'match_objective' or 'first_token_argmax', 'full_answer_preference'"},
    )
    max_length: int = field(default=512, metadata={"help": "Max sequence length for tokenization"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    log_every: int = field(default=1, metadata={"help": "Log frequency (steps)"})
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save projection and config"},
    )
    expid: str = field(default="projection-training", metadata={"help": "Experiment ID for logging"})
    eval_category: Optional[str] = field(
        default="steering",
        metadata={"help": "Evaluation category for logging"},
    )
    output_root: str = field(default=DEFAULT_EVAL_ROOT)
    timestamp: Optional[str] = field(
        default=None,
        metadata={"help": "Timestamp for logging"},
    )
    output_name: Optional[str] = field(
        default=None,
        metadata={"help": "Output name for logging"},
    )
    cache_dir: Optional[str] = field(
        default="./cache_subspace",
        metadata={"help": "Cache directory for filtering results"},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether to use cache when doing the filtering"},
    )
    target_layers: Optional[str] = field(
        default=None,
        metadata={"help": "e.g. 16, or '16,17,18,19' for layer search range"},
    )


def build_filter_cache_path(args: TrainDASArguments, data_path) -> Path:
    model_tag = Path(args.model_name_or_path).name
    if args.lora_path is not None:
        model_tag = f"{Path(args.lora_path).name}"
    dataset_tag = Path(data_path).stem
    image_tag = "w_image" if args.use_image else "wo_image"

    cache_history = "\0".join(
        (
            args.model_name_or_path,
            args.lora_path or "",
            str(Path(data_path).expanduser().resolve()),
            image_tag,
            args.objective,
            args.filter_criterion,
            str(args.max_length),
            str(args.seed),
            # Data order comes from the global RNG stream; don't reuse caches built under per-call seeding.
            "global_rng",
        )
    )
    cache_digest = hashlib.sha256(cache_history.encode("utf-8")).hexdigest()[:12]

    tag = "_".join(
        [
            model_tag,
            image_tag,
            args.objective,
            dataset_tag,
            f"seed{args.seed}",
            cache_digest,
        ]
    )
    tag = re.sub(r"[^A-Za-z0-9_.=-]", "_", tag)
    cache_path = Path(args.cache_dir) / f"subspace_correct_indices_{tag}.pt"
    return cache_path


def get_input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def get_text_layer_path_and_device(model: torch.nn.Module, layer: int) -> Tuple[str, torch.device]:
    # Layout 1: model.model.language_model.layers (Qwen2.5-VL, Gemma4)
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        layer_module = model.model.language_model.layers[layer]
        return f"model.language_model.layers[{layer}].output", next(layer_module.parameters()).device

    # Layout 2: model.language_model.model.layers (e.g. some LoRA-merged checkpoints)
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        layer_module = model.language_model.model.layers[layer]
        return f"language_model.model.layers[{layer}].output", next(layer_module.parameters()).device

    raise ValueError(
        "Cannot locate language model layers. Supported layouts: model.model.language_model.layers, model.language_model.model.layers."
    )


def get_hidden_size(model: torch.nn.Module) -> int:
    if hasattr(model, "config") and hasattr(model.config, "text_config"):
        return int(model.config.text_config.hidden_size)
    if hasattr(model, "model") and hasattr(model.model, "config") and hasattr(model.model.config, "text_config"):
        return int(model.model.config.text_config.hidden_size)
    raise ValueError("Cannot determine text hidden size from model config.")


def _is_gemma(model_name: str) -> bool:
    return "gemma" in model_name.lower()


def _model_slug(model_name: str) -> str:
    """Derive a short filesystem-safe slug from a model name."""
    name = model_name.lower().rsplit("/", 1)[-1]
    return name.replace("-", "_").replace(".", "p")


def _intervened_outputs(intervenable, inputs):
    locations = inputs["sources->base"].squeeze(1)
    if not torch.all(locations == locations[0]):
        raise ValueError("All samples in the batch must have the same intervention location.")
    unit_location_dict = {"sources->base": int(locations[0].item())}

    base_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "mm_token_type_ids": inputs.get("mm_token_type_ids"),
    }
    source_kwargs = {
        "input_ids": inputs["source_input_ids"],
        "attention_mask": inputs["source_attention_mask"],
        "mm_token_type_ids": inputs.get("source_mm_token_type_ids"),
    }
    if inputs.get("pixel_values") is not None:
        base_kwargs["pixel_values"] = inputs["pixel_values"]
        source_kwargs["pixel_values"] = inputs["source_pixel_values"]
        for name in ("image_grid_thw", "image_position_ids"):
            if inputs.get(name) is not None:
                base_kwargs[name] = inputs[name]
                source_kwargs[name] = inputs[f"source_{name}"]

    _, outputs = intervenable(base_kwargs, [source_kwargs], unit_location_dict, subspaces=inputs.get("subspaces"))
    return outputs


def evaluate_full_answer_das(intervenable, val_loader, collator) -> dict[str, float]:
    """Compute SFT-style full-answer preference after source-to-base patching."""

    token_logprobs: dict[tuple[int, bool], list[float]] = defaultdict(list)
    group_metadata: dict[int, tuple[int, bool]] = {}
    source_token_correct = 0
    source_token_total = 0

    with torch.no_grad():
        for raw_inputs in val_loader:
            inputs = collator(raw_inputs)
            labels = inputs["labels"][:, 0]
            outputs = _intervened_outputs(intervenable, inputs)
            logits = outputs.logits[:, -1, :].float()
            logprobs = (
                torch.log_softmax(logits, dim=-1)
                .gather(
                    dim=-1,
                    index=labels.unsqueeze(-1),
                )
                .squeeze(-1)
            )
            predictions = logits.argmax(dim=-1)

            groups = inputs["_preference_group"].reshape(-1)
            candidate_is_source = inputs["_candidate_is_source"].reshape(-1).bool()
            pair_ids = inputs["_pair_id"].reshape(-1)
            source_is_prior = inputs["_source_is_prior"].reshape(-1).bool()
            for row in range(labels.shape[0]):
                group = int(groups[row].item())
                is_source = bool(candidate_is_source[row].item())
                token_logprobs[(group, is_source)].append(float(logprobs[row].item()))
                group_metadata[group] = (
                    int(pair_ids[row].item()),
                    bool(source_is_prior[row].item()),
                )
                if is_source:
                    source_token_correct += int(predictions[row] == labels[row])
                    source_token_total += 1

    context_results: list[bool] = []
    prior_results: list[bool] = []
    pair_results: dict[int, dict[bool, bool]] = defaultdict(dict)
    margins: list[float] = []
    for group, (pair_id, source_is_prior) in group_metadata.items():
        source_scores = token_logprobs.get((group, True), [])
        alternative_scores = token_logprobs.get((group, False), [])
        if not source_scores or not alternative_scores:
            raise ValueError(f"Missing scores for group {group}")
        margin = sum(source_scores) / len(source_scores) - sum(alternative_scores) / len(alternative_scores)
        correct = margin > 0
        margins.append(margin)
        pair_results[pair_id][source_is_prior] = correct
        (prior_results if source_is_prior else context_results).append(correct)

    complete_pairs = [
        directions[False] and directions[True]
        for directions in pair_results.values()
        if False in directions and True in directions
    ]

    def average(values):
        return sum(values) / len(values) if values else 0.0

    context_accuracy = average(context_results)
    prior_accuracy = average(prior_results)
    return {
        "accuracy": average(context_results + prior_results),
        "context_accuracy": context_accuracy,
        "prior_accuracy": prior_accuracy,
        "pair_accuracy": average(complete_pairs),
        "min_accuracy": min(context_accuracy, prior_accuracy),
        "mean_margin": average(margins),
        "source_token_accuracy": source_token_correct / source_token_total if source_token_total else 0.0,
        "n_pairs": len(complete_pairs),
    }


def train_das(
    model: torch.nn.Module,
    processor,
    target_layers,
    train_dataset: Dataset,
    val_dataset: Dataset,
    args: TrainDASArguments,
):
    if args.grad_acc_steps <= 0:
        raise ValueError(f"grad_acc_steps must be >= 1, got {args.grad_acc_steps}")
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError(f"warmup_ratio must be in [0, 1], got {args.warmup_ratio}")
    if not target_layers:
        raise ValueError("target_layer must be provided for DAS training.")

    component_path, layer_device = get_text_layer_path_and_device(model, target_layers[0])
    hidden_size = get_hidden_size(model)

    logger.info("Preparing DAS intervention at layer %d (%s)", target_layers[0], component_path)
    projection = LowRankOrthogonalProjection(
        embed_dim=hidden_size,
        rank=args.das_rank,
    )

    class InterchangeIntervention(TrainableIntervention, DistributedRepresentationIntervention):
        def __init__(
            self,
            projection: LowRankOrthogonalProjection,
            layer,
            device=None,
            last_token_only=True,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.proj = projection
            self.layer = layer
            self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.last_token_only = last_token_only
            self.proj.to(self.device)
            self._activated = False
            self._source = False

        def forward(self, target, source, subspaces=None):
            # h_t = (I-P) h_t + P h_s
            intervened_target = self.proj(source, target)
            return intervened_target.to(target.dtype)

    intervention_type = "block_output"
    # layer = 16

    config = IntervenableConfig(
        [
            {
                "layer": layer_idx,
                "component": intervention_type,
                "intervention_type": lambda **kwargs: InterchangeIntervention(projection, **kwargs),
            }
            for layer_idx in target_layers
        ]
    )

    intervenable = IntervenableModel(config, model)
    intervenable.disable_model_gradients()

    key = list(intervenable.interventions.keys())[0]
    try:
        intervenable.set_device(layer_device)
    except Exception as e:
        # TO DEVICE for 4bit
        for k in intervenable.interventions:
            intervenable.interventions[k][0].to(layer_device)
            intervenable.interventions[k][0].rotate_layer.to(layer_device)
    input_device = get_input_device(model)
    full_answer = args.objective == "full_answer"
    full_answer_collator = (
        partial(full_answer_das_collator, pad_token_id=processor.tokenizer.pad_token_id) if full_answer else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=full_answer,
        drop_last=False,
        collate_fn=full_answer_collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=full_answer_collator,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_acc_steps)
    total_train_steps = max(1, updates_per_epoch * args.das_epochs)
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    seen = set()
    unique_params = []
    for p in intervenable.get_trainable_parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            unique_params.append(p)

    optimizer = torch.optim.AdamW(
        unique_params,
        lr=args.das_lr,
        weight_decay=0,  # Must be 0 or use Adam instead of AdamW
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    collator = partial(inputs_collator, device=input_device)
    metrics_fn = partial(compute_metrics, verbose=False, tokenizer=processor.tokenizer)

    logger.info("Training DAS via IntervenableModel.train_alignment ...")
    intervenable.train_alignment(
        processor=processor,
        train_dataloader=train_loader,
        compute_loss=compute_loss,
        compute_metrics=metrics_fn,
        inputs_collator=collator,
        epochs=args.das_epochs,
        lr=args.das_lr,
        gradient_accumulation_steps=args.grad_acc_steps,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
    )

    # logger.info("Evaluating DAS via IntervenableModel.eval_alignment on train split ...")
    # train_metrics = intervenable.eval_alignment(
    #     eval_dataloader=train_loader,
    #     compute_metrics=metrics_fn,
    #     inputs_collator=collator,
    # )
    train_metrics = None
    logger.info("Evaluating DAS via IntervenableModel.eval_alignment on val split ...")
    if full_answer:
        val_metrics = evaluate_full_answer_das(intervenable, val_loader, collator)
    else:
        val_metrics = intervenable.eval_alignment(
            eval_dataloader=val_loader,
            compute_metrics=metrics_fn,
            inputs_collator=collator,
        )

    trained_projection = intervenable.interventions[key][0]
    return trained_projection, train_metrics, val_metrics


def main() -> None:
    parser = HfArgumentParser((TrainDASArguments,))
    (args,) = parser.parse_args_into_dataclasses()

    if args.objective not in ("full_answer", "first_token"):
        raise ValueError(f"Invalid objective: {args.objective}. Must be 'full_answer' or 'first_token'.")

    if args.filter_criterion not in ("match_objective", "first_token_argmax", "full_answer_preference"):
        raise ValueError(f"Invalid filter_criterion: {args.filter_criterion}.")

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"max_samples must be positive or None, got {args.max_samples}")

    eval_run = prepare_eval_run(
        category=args.eval_category,
        expid=args.expid,
        output_root=args.output_root,
        timestamp=args.timestamp,
        output_name=args.output_name,
    )

    configure_run_logging(eval_run, kind="das_projection_training")

    setup_seed(args.seed)
    runtime_device = get_device()
    logger.info("Runtime device: %s", runtime_device)

    model, processor = load_model_and_processor(
        args.model_name_or_path,
        lora_path=args.lora_path,
        merge_lora=True,
        torch_dtype=get_torch_dtype(runtime_device),
        use_cache=False,
    )

    n_layers = get_num_layers(model)
    logger.info("Model has %d text layers", n_layers)

    use_image = args.use_image
    logger.info("Use images: %s", use_image)

    logger.info("Loading dataset from: %s", args.train_data)
    train_context_meta, train_prior_meta, train_data = prepare_vlm_patching_data(
        processor=processor,
        dataset_path=args.train_data,
        image_root=args.image_root,
        device=runtime_device,
        use_image=use_image,
    )
    val_context_meta, val_prior_meta, val_data = prepare_vlm_patching_data(
        processor=processor,
        dataset_path=args.val_data,
        image_root=args.image_root,
        device=runtime_device,
        use_image=use_image,
    )
    # # 444, 450
    # logger.info("Loaded %d paired samples", train_data.source_input_ids.shape[0])
    # this will reshape the pixel values to (B, patches, dim) for train dataset.
    logger.info("Filtered pairs where model is correct on both prompts...")
    train_cache_path = build_filter_cache_path(args, args.train_data) if args.use_cache else None
    if train_cache_path is not None and train_cache_path.exists():
        logger.info("Loading cached correct indices from: %s", train_cache_path)

    filter_counts = None
    if args.objective == "full_answer":
        train_source_answers = candidate_token_ids(
            processor,
            [metadata.prior_answer for metadata in train_prior_meta],
            [metadata.response_prefix for metadata in train_prior_meta],
        )
        train_target_answers = candidate_token_ids(
            processor,
            [metadata.context_answer for metadata in train_context_meta],
            [metadata.response_prefix for metadata in train_context_meta],
        )
        if args.filter_criterion == "first_token_argmax":
            source_first_tokens = train_data.source_answer_index.detach().cpu().tolist()
            target_first_tokens = train_data.target_answer_index.detach().cpu().tolist()
            if [tokens[0] for tokens in train_source_answers] != source_first_tokens:
                raise ValueError("Prior complete-answer tokens do not align with first-token labels")
            if [tokens[0] for tokens in train_target_answers] != target_first_tokens:
                raise ValueError("Context complete-answer tokens do not align with first-token labels")
            train_data, filter_counts, selected_indices = filter_by_correct_predictions(
                model=model,
                data=train_data,
                batch_size=args.filter_batch_size,
                max_samples=args.max_samples,
                processor=processor,
                cache_path=train_cache_path,
                require_distinct_answer_tokens=False,
                require_distinct_answer_initials=True,
                source_answer_texts=[metadata.prior_answer for metadata in train_prior_meta],
                target_answer_texts=[metadata.context_answer for metadata in train_context_meta],
                return_filter_stats=True,
                return_selected_indices=True,
            )
            train_source_answers = [train_source_answers[i] for i in selected_indices]
            train_target_answers = [train_target_answers[i] for i in selected_indices]
        else:
            train_data, train_source_answers, train_target_answers, filter_counts = filter_by_full_answer_preferences(
                model=model,
                processor=processor,
                data=train_data,
                source_answer_ids=train_source_answers,
                target_answer_ids=train_target_answers,
                batch_size=args.filter_batch_size,
                max_samples=args.max_samples,
                max_length=args.max_length,
                cache_path=train_cache_path,
                require_distinct_answer_tokens=True,
                return_filter_stats=True,
            )
        num_correct_pairs = filter_counts["eligible_intersection_pairs"]
        val_source_answers = candidate_token_ids(
            processor,
            [metadata.prior_answer for metadata in val_prior_meta],
            [metadata.response_prefix for metadata in val_prior_meta],
        )
        val_target_answers = candidate_token_ids(
            processor,
            [metadata.context_answer for metadata in val_context_meta],
            [metadata.response_prefix for metadata in val_context_meta],
        )
        train_dataset = build_full_answer_das_dataset(
            train_data,
            train_source_answers,
            train_target_answers,
        )
        train_pair_count = len(train_source_answers)
        if not args.filter_only:
            val_dataset = build_full_answer_das_dataset(
                val_data,
                val_source_answers,
                val_target_answers,
                include_alternatives=True,
            )
    else:
        train_data, filter_counts = filter_by_correct_predictions(
            model=model,
            data=train_data,
            batch_size=args.filter_batch_size,
            max_samples=args.max_samples,
            processor=processor,
            cache_path=train_cache_path,
            require_distinct_answer_tokens=True,
            return_filter_stats=True,
        )
        num_correct_pairs = filter_counts["eligible_intersection_pairs"]
        train_dataset = build_das_datasets(data=train_data)
        train_pair_count = len(train_dataset) // 2
        if not args.filter_only:
            val_dataset = build_das_datasets(data=val_data)

    if args.filter_only:
        enough_pairs = args.max_samples is None or num_correct_pairs >= args.max_samples
        logger.info(
            "Correctly predicted pairs: %d | selected: %s | requested: %s | enough: %s",
            num_correct_pairs,
            len(train_context_meta),
            train_pair_count,
            args.max_samples,
            enough_pairs,
        )
        write_json_result(
            eval_run,
            {
                "model_name_or_path": args.model_name_or_path,
                "lora_path": args.lora_path,
                "train_data": args.train_data,
                "seed": args.seed,
                "objective": args.objective,
                "filter_criterion": args.filter_criterion,
                "max_samples": args.max_samples,
                "total_training_pairs": len(train_context_meta),
                "correct_training_pairs": num_correct_pairs,
                "selected_training_pairs": train_pair_count,
                "enough_requested_pairs": enough_pairs,
                "filter_counts": filter_counts,
            },
            kind="das_correct_pair_audit",
            metadata={
                "model_name_or_path": args.model_name_or_path,
                "use_image": args.use_image,
            },
        )
        return

    logger.info(
        "Train pairs: %d | Train token rows: %d | Val pairs:  %d | Val token rows:  %d",
        train_pair_count,
        len(train_dataset),
        len(val_context_meta),
        len(val_dataset),
    )

    target_layers = [int(x) for x in args.target_layers.split(",")]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        trained_projection, train_metrics, val_metrics = train_das(
            model=model,
            processor=processor,
            target_layers=target_layers,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
        )
    output_dir = (
        Path(f"{args.output_dir}_layer_{args.target_layers}")
        if args.output_dir
        else eval_run.artifact_dir(f"projection_layer_{args.target_layers}")
    )
    trained_projection.proj.save_pretrained(output_dir)
    write_json_result(
        eval_run,
        {
            "projection_path": str(output_dir),
            "train_data": args.train_data,
            "val_data": args.val_data,
            "target_layers": target_layers,
            "seed": args.seed,
            "objective": args.objective,
            "filter_criterion": args.filter_criterion,
            "das_rank": args.das_rank,
            "das_epochs": args.das_epochs,
            "das_lr": args.das_lr,
            "warmup_ratio": args.warmup_ratio,
            "gradient_accumulation_steps": args.grad_acc_steps,
            "max_samples": args.max_samples,
            "correct_training_pairs": num_correct_pairs,
            "train_pairs": train_pair_count,
            "enough_requested_pairs": (args.max_samples is None or num_correct_pairs >= args.max_samples),
            "filter_counts": filter_counts,
            "train_token_rows": len(train_dataset),
            "validation_pairs": len(val_context_meta),
            "validation_token_rows": len(val_dataset),
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
        },
        kind="das_projection_training",
        metadata={"model_name_or_path": args.model_name_or_path, "use_image": args.use_image},
    )


if __name__ == "__main__":
    main()
