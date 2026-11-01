import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    HfArgumentParser,
)
from trl import SFTConfig, SFTTrainer

from src.checkpoint_selection import JointGainPairTieBreakMixin
from src.eval_results import (
    DEFAULT_EVAL_ROOT,
    configure_run_logging,
    prepare_eval_run,
    write_json_result,
)
from src.model_utils import get_device, load_model_and_processor, setup_seed
from src.sft.data_process import (
    DEFAULT_TEST_DATASETS,
    DEFAULT_VALIDATION_DATASETS,
    CustomCollator,
    Task,
    TaskMixture,
    load_jsonl,
)
from src.sft.generation_eval import evaluate_exact_match_generation
from src.sft.preference_eval import (
    PREFERENCE_LABEL_NAMES,
    BaselineRelativePreferenceMetrics,
    PreferenceDataset,
    ValidationSuitePreferenceMetrics,
    preprocess_logits_for_preference,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "HuggingFace model id or local path"},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit quantisation via BitsAndBytes"},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit quantisation via BitsAndBytes"},
    )
    # LoRA hyper-parameters
    lora_r: int = field(default=32, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Which linear modules to apply LoRA to"},
    )
    lora_attention_types: Optional[List[str]] = field(
        default=None, metadata={"help": "Optional decoder attention type filter, such as full_attention."}
    )
    # Which parts of the model to finetune
    finetune_llm: bool = field(default=True, metadata={"help": "Apply LoRA to LLM language model layers"})
    finetune_vision: bool = field(default=False, metadata={"help": "Apply LoRA to vision encoder layers"})


@dataclass
class DataArguments:
    dataset_paths: str = field(
        default="dataset/whatifvis/spatial_temporal_train.jsonl",
        metadata={"help": "Comma-separated paths to training JSONL files"},
    )
    image_root: Optional[str] = field(
        default="dataset/whatifvis",
        metadata={"help": "Optional root directory to prepend to image paths"},
    )
    validation_dataset_path: Optional[str] = field(
        default=DEFAULT_VALIDATION_DATASETS,
        metadata={"help": "Comma-separated validation JSONLs evaluated during training"},
    )
    validation_image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Validation image root; defaults to image_root"},
    )
    validation_stop: Optional[int] = field(
        default=None,
        metadata={"help": "Optional validation sample limit for debugging"},
    )
    validation_max_examples_per_dataset: Optional[int] = field(
        default=65,
        metadata={
            "help": (
                "Keep the first N raw examples (file order; each gives a context and a prior prompt) "
                "of every validation JSONL. Matches the value sweep's max_rows_per_dataset. Use a value above the set size (e.g. 100000) to keep all."
            )
        },
    )
    test_dataset_paths: Optional[str] = field(
        default=DEFAULT_TEST_DATASETS,
        metadata={"help": ("Comma-separated held-out test JSONLs evaluated once after training")},
    )
    test_image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Test image root; defaults to image_root"},
    )
    test_stop: Optional[int] = field(
        default=None,
        metadata={"help": "Optional test sample limit for debugging"},
    )
    test_generation_em: bool = field(
        default=True,
        metadata={"help": "Run generation exact-match on final test splits"},
    )
    test_generation_max_new_tokens: int = field(
        default=16,
        metadata={"help": "Max new tokens for generation exact-match evaluation"},
    )
    stop: Optional[int] = field(
        default=None,
        metadata={"help": "Optional sample limit for debugging"},
    )
    use_image: bool = field(
        default=True,
        metadata={"help": "Whether to include image features in the training data"},
    )


@dataclass
class ExperimentArguments:
    expid: Optional[str] = field(
        default=None,
        metadata={"help": "Experiment id; defaults to the checkpoint directory name"},
    )
    eval_category: str = field(default="sft")
    output_root: str = field(default=DEFAULT_EVAL_ROOT)
    timestamp: Optional[str] = field(default=None)
    output_name: str = field(default="training.json")


@dataclass
class TrainingArguments(SFTConfig):
    """TRL training arguments with project-compatible defaults.

    This is passed directly to :class:`SFTTrainer`, so every inherited CLI
    option has the value that the trainer actually uses.
    """

    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum tokenised sequence length"},
    )
    remove_unused_columns: bool = field(default=False)
    dataset_kwargs: Optional[dict] = field(default_factory=lambda: {"skip_prepare_dataset": True})
    bf16: Optional[bool] = field(default=True)
    gradient_checkpointing: bool = field(default=False)
    eval_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    save_total_limit: Optional[int] = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: Optional[str] = field(default="eval_pair_accuracy")
    greater_is_better: Optional[bool] = field(default=True)
    normalize_stop_loss_by_token_count: bool = field(
        default=False,
        metadata={
            "help": (
                "For pure SFT, optimize mean answer-token CE plus mean trailing "
                "stop/format-token CE instead of averaging all response tokens "
                "together"
            )
        },
    )
    stop_loss_weight: float = field(
        default=1.0,
        metadata={"help": ("Weight for the stop/format-token CE loss when normalize_stop_loss_by_token_count is True")},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Overwrite the output directory if it exists"},
    )


def dataset_entry_identity(entry):
    image_name = Path(entry.image or "").name.lower().replace(" ", "_")

    def normalize(value):
        return " ".join(value.casefold().split())

    return image_name, normalize(entry.question), normalize(entry.context_answer), normalize(entry.prior_answer)


def dataset_overlap_counts(source_paths, heldout_paths):
    source_keys = {
        source_path: {dataset_entry_identity(entry) for entry in load_jsonl(source_path)}
        for source_path in source_paths
    }
    overlaps = {}
    for heldout_path in heldout_paths:
        heldout_entries = load_jsonl(heldout_path)
        for source_path, keys in source_keys.items():
            count = sum(dataset_entry_identity(entry) in keys for entry in heldout_entries)
            if count:
                overlaps[(source_path, heldout_path)] = count
    return overlaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_quantization_config(load_in_4bit: bool, load_in_8bit: bool) -> Optional[BitsAndBytesConfig]:
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def build_lora_config(
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Optional[List[str]],
    model: Optional[torch.nn.Module] = None,
    finetune_llm: bool = True,
    finetune_vision: bool = False,
    attention_types: Optional[List[str]] = None,
) -> LoraConfig:
    if model is not None:
        target_modules = select_lora_target_modules(
            model,
            patterns=target_modules,
            include_language=finetune_llm,
            include_vision=finetune_vision,
            attention_types=attention_types,
        )
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(target_modules or []),
    )


def _filter_language_modules(
    model: torch.nn.Module,
    patterns: Optional[List[str]] = None,  # kept for compatibility, now ignored
):
    """
    Return the names of all linear layers inside the language_model subtree.

    This is a modified version that no longer restricts results to q/k/v/o
    projections, every nn.Linear under ``language_model`` is included.
    """
    language_modules = []

    for name, module in model.named_modules():
        if "language_model" not in name:
            continue
        if "qwen" in model.__class__.__name__.lower():
            if isinstance(module, nn.Linear):
                if "mlp" in name.lower():
                    continue
                language_modules.append(name)
        else:
            if isinstance(module, nn.Linear):
                if (
                    "mlp" in name.lower()
                    or "per_layer_model_projection" in name.lower()
                    or "per_layer_projection" in name.lower()
                    or "per_layer_input_gate" in name.lower()
                ):
                    continue
                language_modules.append(name)
    if not language_modules:
        raise ValueError("No linear modules found in language_model layers.")

    return _filter_requested_modules(language_modules, patterns)


def _filter_requested_modules(candidates, patterns):
    if not patterns:
        return candidates
    requested = [value.strip() for pattern in patterns for value in pattern.split(",") if value.strip()]
    if not requested:
        raise ValueError("No valid module patterns specified in lora_target_modules")

    matched = [
        name for name in candidates if any(name == pattern or name.endswith(f".{pattern}") for pattern in requested)
    ]
    missing = [
        pattern
        for pattern in requested
        if not any(name == pattern or name.endswith(f".{pattern}") for name in candidates)
    ]
    if missing:
        raise ValueError(
            f"Requested LoRA modules not found in model: {missing}. Available modules: {sorted(candidates)}"
        )
    return matched


def _filter_vision_modules(model, patterns=None):
    vision_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and ({"visual", "vision_tower"} & set(name.lower().split(".")))
    ]
    if not vision_modules:
        raise ValueError("No linear modules found in vision_tower or visual layers.")
    return _filter_requested_modules(vision_modules, patterns)


def _filter_language_attention_types(model, candidates, attention_types):

    if attention_types is None:
        return candidates

    requested = {
        value.strip() for attention_type in attention_types for value in attention_type.split(",") if value.strip()
    }

    if not requested:
        return ValueError("No valid attention types specified in lora_attention_types")

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)

    if not layer_types:
        raise ValueError("Model does not have layer_types in its text_config; cannot filter by attention type.")

    available = set(layer_types)
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"Unknown attention types requested: {unknown}. Available types: {sorted(available)}")

    selected_layers = {index for index, attention_type in enumerate(layer_types) if attention_type in requested}
    layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
    selected = []
    for name in candidates:
        match = layer_pattern.search(name)
        if match and int(match.group(1)) in selected_layers:
            selected.append(name)
    if not selected:
        raise ValueError("No language-model LoRA modules matched attention types: " + ", ".join(sorted(requested)))
    return selected


def select_lora_target_modules(
    model: torch.nn.Module,
    *,
    patterns: Optional[List[str]],
    include_language: bool,
    include_vision: bool,
    attention_types: Optional[List[str]] = None,
) -> List[str]:
    """Resolve the exact LoRA targets for the selected model components."""

    if not include_language and not include_vision:
        raise ValueError("At least one of --finetune_llm or --finetune_vision must be true")
    if attention_types is not None and not include_language:
        raise ValueError("lora_attention_types requires --finetune_llm true")

    language_candidates = _filter_language_modules(model) if include_language else []
    language_candidates = _filter_language_attention_types(
        model,
        language_candidates,
        attention_types,
    )
    vision_candidates = _filter_vision_modules(model) if include_vision else []
    candidates = language_candidates + vision_candidates
    candidates = list(dict.fromkeys(candidates))
    selected = _filter_requested_modules(candidates, patterns)

    if include_language and not set(selected).intersection(language_candidates):
        raise ValueError("lora_target_modules selects no language-model modules")
    if include_vision and not set(selected).intersection(vision_candidates):
        raise ValueError("lora_target_modules selects no vision-tower modules")
    return selected


def filter_layers_with_complete_projections(lst):
    # Step 1: Identify all layer indices that have all four projections
    layer_projections = defaultdict(set)  # layer -> set of projections found

    # Pattern to extract layer number and projection suffix from self_attn entries
    pattern = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)")

    for item in lst:
        match = pattern.search(item)
        if match:
            layer = int(match.group(1))
            proj = match.group(2)
            layer_projections[layer].add(proj)

    # Layers that have all four
    complete_layers = {
        layer for layer, projs in layer_projections.items() if {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(projs)
    }

    # Step 2: Keep all elements whose layer (if any) is in complete_layers
    # Extract layer number from any element (self_attn or not) using pattern that looks for 'layers.<number>'
    layer_pattern = re.compile(r"layers\.(\d+)")

    filtered = []
    for item in lst:
        match = layer_pattern.search(item)
        if not match:
            # No layer number found? Keep it? According to rule, we only filter by layer, so keep if no layer.
            # But in typical model names, there will be a layer. For safety, we keep.
            filtered.append(item)
        else:
            layer = int(match.group(1))
            if layer in complete_layers:
                filtered.append(item)
            # else: drop

    return filtered


class SequentialSFTTrainer(JointGainPairTieBreakMixin, SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_normalized_loss_sums = defaultdict(float)
        self._stop_normalized_loss_steps = 0

    def _get_train_sampler(self, dataset=None):
        # Override to use a sequential sampler instead of random sampling
        if dataset is None:
            dataset = self.train_dataset
        return torch.utils.data.SequentialSampler(dataset)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        inputs = dict(inputs)
        for key in PREFERENCE_LABEL_NAMES[1:]:
            inputs.pop(key, None)
        answer_token_mask = inputs.pop("_answer_token_mask", None)
        stop_token_mask = inputs.pop("_stop_token_mask", None)
        if self.args.normalize_stop_loss_by_token_count and model.training:
            if answer_token_mask is None or stop_token_mask is None:
                raise ValueError("Stop-normalized SFT requires answer and stop token masks")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss, answer_loss, stop_loss = answer_stop_normalized_cross_entropy(
                outputs.logits,
                labels,
                answer_token_mask,
                stop_token_mask,
                stop_loss_weight=self.args.stop_loss_weight,
            )
            self._stop_normalized_loss_sums["answer_loss"] += float(answer_loss.detach())
            self._stop_normalized_loss_sums["stop_loss"] += float(stop_loss.detach())
            self._stop_normalized_loss_steps += 1
            return (loss, outputs) if return_outputs else loss
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        if self._stop_normalized_loss_steps and "loss" in logs:
            for key, total in self._stop_normalized_loss_sums.items():
                logs[key] = total / self._stop_normalized_loss_steps
            self._stop_normalized_loss_sums.clear()
            self._stop_normalized_loss_steps = 0
        super().log(logs, start_time)


def answer_stop_normalized_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    answer_token_mask: torch.Tensor,
    stop_token_mask: torch.Tensor,
    *,
    stop_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted average of mean answer CE and mean trailing stop-token CE.

    Dividing by ``1 + stop_loss_weight`` keeps the loss scale (and so the
    gradient-clipping behaviour) independent of the stop weight.
    """

    if stop_loss_weight < 0:
        raise ValueError("stop_loss_weight must be non-negative")
    if answer_token_mask.shape != labels.shape:
        raise ValueError("answer_token_mask and labels must have the same shape")
    if stop_token_mask.shape != labels.shape:
        raise ValueError("stop_token_mask and labels must have the same shape")

    supervised_mask = labels[:, 1:].ne(-100)
    answer_mask = answer_token_mask[:, 1:].to(dtype=torch.bool) & supervised_mask
    stop_mask = stop_token_mask[:, 1:].to(dtype=torch.bool) & supervised_mask
    if bool((answer_mask & stop_mask).any()):
        raise ValueError("Answer and stop token masks must not overlap")
    if not torch.equal(answer_mask | stop_mask, supervised_mask):
        raise ValueError("Answer and stop token masks must cover every labeled token")
    if not bool(answer_mask.any()):
        raise ValueError("Stop-normalized SFT requires at least one answer token")
    if not bool(stop_mask.any()):
        raise ValueError("Stop-normalized SFT requires at least one stop token")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    answer_loss = (
        F.cross_entropy(
            shifted_logits[answer_mask].float(),
            shifted_labels[answer_mask],
            reduction="sum",
        )
        / answer_mask.sum()
    )
    stop_loss = (
        F.cross_entropy(
            shifted_logits[stop_mask].float(),
            shifted_labels[stop_mask],
            reduction="sum",
        )
        / stop_mask.sum()
    )
    return (answer_loss + stop_loss_weight * stop_loss) / (1 + stop_loss_weight), answer_loss, stop_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ExperimentArguments))
    model_args, data_args, training_args, experiment_args = parser.parse_args_into_dataclasses()
    checkpoint_dir = Path(training_args.output_dir)
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint output is not a directory: {checkpoint_dir}")
    if (
        checkpoint_dir.exists()
        and any(checkpoint_dir.iterdir())
        and not training_args.overwrite_output_dir
        and not training_args.resume_from_checkpoint
    ):
        raise FileExistsError(
            f"Checkpoint directory is not empty: {checkpoint_dir}. "
            "Choose a new --output_dir or pass --overwrite_output_dir true."
        )

    expid = experiment_args.expid or Path(training_args.output_dir).name
    run = prepare_eval_run(
        category=experiment_args.eval_category,
        expid=expid,
        output_root=experiment_args.output_root,
        timestamp=experiment_args.timestamp,
        output_name=experiment_args.output_name,
    )
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        configure_run_logging(run, kind="sft-training")

    # Seed before model/adapter initialization; honor the trainer's optional
    # full-determinism setting instead of claiming deterministic behavior.
    setup_seed(training_args.seed, deterministic=training_args.full_determinism)

    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Seed: {training_args.seed}")
    os.environ.setdefault("WANDB_DIR", str(run.run_dir / "wandb"))
    if training_args.run_name is None:
        training_args.run_name = f"{Path(training_args.output_dir).name}_{run.timestamp}"

    dataset_paths = [p.strip() for p in data_args.dataset_paths.split(",") if p.strip()]
    if not dataset_paths:
        raise ValueError("No dataset paths provided")

    logger.info(f"Training on {len(dataset_paths)} dataset(s): {dataset_paths}")
    validation_dataset_paths = [p.strip() for p in (data_args.validation_dataset_path or "").split(",") if p.strip()]

    test_dataset_paths = [path.strip() for path in (data_args.test_dataset_paths or "").split(",") if path.strip()]

    heldout_paths = validation_dataset_paths + test_dataset_paths
    training_overlaps = dataset_overlap_counts(dataset_paths, heldout_paths)
    if training_overlaps:
        details = ", ".join(f"{source} -> {heldout}: {count}" for (source, heldout), count in training_overlaps.items())
        raise ValueError(f"Training data overlaps held-out data: {details}")
    if validation_dataset_paths:
        logger.info("Validation datasets: %s", validation_dataset_paths)
    if test_dataset_paths:
        logger.info("Final test datasets: %s", test_dataset_paths)
    if training_args.normalize_stop_loss_by_token_count:
        logger.info(
            "Pure-SFT loss: (mean answer-token CE + w * mean stop/format-token CE) / (1 + w), w=%.4f",
            training_args.stop_loss_weight,
        )

    # ------------------------------------------------------------------
    # Model & processor
    # ------------------------------------------------------------------
    bnb_config = build_quantization_config(model_args.load_in_4bit, model_args.load_in_8bit)

    model, processor = load_model_and_processor(
        model_args.model_name_or_path,
        device_map="auto",
        quantization_config=bnb_config,
        padding_side="right",
        eval_mode=False,
    )

    if bnb_config is None:
        model = model.to(device)

    # ------------------------------------------------------------------
    # Freeze & LoRA configuration
    # ------------------------------------------------------------------
    lora_config = build_lora_config(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=model_args.lora_target_modules,
        model=model,
        finetune_llm=model_args.finetune_llm,
        finetune_vision=model_args.finetune_vision,
        attention_types=model_args.lora_attention_types,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # ------------------------------------------------------------------
    # Dataset & collator
    # ------------------------------------------------------------------
    tasks: List[Task] = []
    for dataset_path in dataset_paths:
        task = Task(
            processor=processor,
            max_length=training_args.max_length,
            jsonl_path=dataset_path,
            image_root=data_args.image_root,
            stop=data_args.stop,
            training=True,
            use_image=data_args.use_image,
            shuffle=True,
            normalize_stop_loss_by_token_count=training_args.normalize_stop_loss_by_token_count,
        )
        tasks.append(task)
        logger.info(f"  [{Path(dataset_path).stem}] Loaded {len(task)} samples")

    dataset = TaskMixture(
        tasks=tasks,
        shuffle=False,
    )
    if not dataset:
        raise ValueError("No training samples found in the provided datasets")

    validation_dataset = None
    validation_split_prompt_counts = []
    if validation_dataset_paths:
        max_val_examples = data_args.validation_max_examples_per_dataset
        if max_val_examples is not None and max_val_examples <= 0:
            raise ValueError("validation_max_examples_per_dataset must be positive or None")
        # Task.stop counts prompts (two per raw example: context, prior).
        validation_prompt_stops = [
            stop for stop in (data_args.validation_stop, None if max_val_examples is None else 2 * max_val_examples)
            if stop is not None
        ]
        validation_prompt_stop = min(validation_prompt_stops) if validation_prompt_stops else None
        validation_tasks: List[Task] = []
        for validation_dataset_path in validation_dataset_paths:
            validation_task = Task(
                processor=processor,
                max_length=training_args.max_length,
                jsonl_path=validation_dataset_path,
                image_root=data_args.validation_image_root or data_args.image_root,
                stop=validation_prompt_stop,
                training=False,
                use_image=data_args.use_image,
            )
            if max_val_examples is not None and len(validation_task) < 2 * max_val_examples:
                logger.warning(
                    "Validation set %s has only %d prompts (< %d requested examples x 2)",
                    validation_dataset_path,
                    len(validation_task),
                    max_val_examples,
                )
            logger.info(
                "  [%s] Using %d validation prompts (%d examples)",
                Path(validation_dataset_path).stem,
                len(validation_task),
                len(validation_task) // 2,
            )
            validation_tasks.append(validation_task)
            validation_split_prompt_counts.append((Path(validation_dataset_path).stem, len(validation_task)))
        logger.info(f"  [{Path(dataset_path).stem}] Loaded {len(validation_tasks)} validation datasets")

        validation_prompts = TaskMixture(
            tasks=validation_tasks,
            shuffle=False,
        )
        validation_dataset = PreferenceDataset(
            prompt_dataset=validation_prompts,
            processor=processor,
            max_length=training_args.max_length,
        )

    collator = CustomCollator(processor=processor, max_length=training_args.max_length)

    # Log one tokenized sample as a sanity check
    sample_1, sample_2 = dataset[0], dataset[1]
    encoded = collator([sample_1, sample_2])
    logger.info(
        f"Tokenized samples: \n{processor.tokenizer.decode(encoded['input_ids'][0])}\n\n{processor.tokenizer.decode(encoded['input_ids'][1])}"
    )
    logger.info(f"Training samples: {len(dataset)}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    if validation_dataset is not None or test_dataset_paths:
        training_args.label_names = list(PREFERENCE_LABEL_NAMES)
    preference_metrics = (
        ValidationSuitePreferenceMetrics(validation_split_prompt_counts)
        if len(validation_split_prompt_counts) > 1
        else BaselineRelativePreferenceMetrics()
    )
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "eval_dataset": validation_dataset,
        "data_collator": collator,
        "args": training_args,
        "compute_metrics": preference_metrics,
        "preprocess_logits_for_metrics": preprocess_logits_for_preference,
    }
    trainer = SequentialSFTTrainer(**trainer_kwargs)

    baseline_validation_metrics = {}
    selection_metric = training_args.metric_for_best_model or ""
    if selection_metric.removeprefix("eval_") == "joint_gain":
        if validation_dataset is None:
            raise ValueError("joint_gain checkpoint selection requires validation data")
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        disable_adapter = getattr(unwrapped_model, "disable_adapter", None)
        if disable_adapter is None:
            raise TypeError("joint_gain checkpoint selection requires a PEFT model with disable_adapter()")
        with disable_adapter():
            baseline_validation_metrics = trainer.evaluate(
                eval_dataset=validation_dataset,
                metric_key_prefix="base_val",
            )
        preference_metrics.set_baseline(
            context_accuracy=baseline_validation_metrics["base_val_context_accuracy"],
            prior_accuracy=baseline_validation_metrics["base_val_prior_accuracy"],
        )
        logger.info(
            "Validation baseline: context=%.4f prior=%.4f pair=%.4f; "
            "selecting checkpoints by min(context_gain, prior_gain), then "
            "pair_accuracy on ties",
            baseline_validation_metrics["base_val_context_accuracy"],
            baseline_validation_metrics["base_val_prior_accuracy"],
            baseline_validation_metrics["base_val_pair_accuracy"],
        )

    train_output = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    logger.info(
        "Best checkpoint: %s (%s=%s)",
        trainer.state.best_model_checkpoint,
        training_args.metric_for_best_model,
        trainer.state.best_metric,
    )
    validation_metrics = trainer.evaluate(eval_dataset=validation_dataset) if validation_dataset is not None else {}

    # Do not construct or score the held-out test split until model selection
    # has finished and Trainer has restored the best validation checkpoint.
    preference_metrics.clear_baseline()
    trainer.compute_metrics = BaselineRelativePreferenceMetrics()
    test_metrics = {}
    test_tasks: List[Task] = []
    for test_dataset_path in test_dataset_paths:
        test_task = Task(
            processor=processor,
            max_length=training_args.max_length,
            jsonl_path=test_dataset_path,
            image_root=data_args.test_image_root or data_args.image_root,
            stop=data_args.test_stop,
            training=False,
            use_image=data_args.use_image,
        )
        test_tasks.append(test_task)
        test_prompts = TaskMixture(
            tasks=[test_task],
            shuffle=False,
        )
        if not test_prompts:
            raise ValueError("Test dataset is empty")
        test_dataset = PreferenceDataset(
            test_prompts,
            processor,
            max_length=training_args.max_length,
        )
        logger.info(
            " [test:%s] Loaded %d prompts (%d candidate rows)",
            Path(test_dataset_path).stem,
            len(test_prompts),
            len(test_dataset),
        )
        dataset_name = Path(test_dataset_path).stem
        test_metrics[dataset_name] = trainer.evaluate(
            eval_dataset=test_dataset,
            metric_key_prefix=f"test_{dataset_name}",
        )

    test_em_metrics = {}
    if data_args.test_generation_em and test_tasks:
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            test_em_metrics = evaluate_exact_match_generation(
                model=trainer.accelerator.unwrap_model(trainer.model),
                processor=processor,
                dataset=TaskMixture(tasks=test_tasks, shuffle=False),
                device=device,
                batch_size=training_args.per_device_eval_batch_size,
                max_length=training_args.max_length,
                max_new_tokens=data_args.test_generation_max_new_tokens,
                num_workers=training_args.dataloader_num_workers,
                prefetch_factor=training_args.dataloader_prefetch_factor,
            )
            for dataset_name, metrics in test_em_metrics["per_dataset"].items():
                logger.info(
                    " [test:%s] EM context=%.4f prior=%.4f pair=%.4f",
                    dataset_name,
                    metrics["context_accuracy"],
                    metrics["prior_accuracy"],
                    metrics["pair_accuracy"],
                )
        trainer.accelerator.wait_for_everyone()

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------
    trainer.save_model(training_args.output_dir)
    if trainer.is_world_process_zero():
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        processor.save_pretrained(str(output_dir))
        logger.info(f"Model saved to {output_dir}")
        write_json_result(
            run,
            {
                "checkpoint_path": str(output_dir),
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "metrics": train_output.metrics,
                "validation_metrics": validation_metrics,
                "baseline_validation_metrics": baseline_validation_metrics,
                "test_metrics": test_metrics,
                "test_em_metrics": test_em_metrics,
                "config": {
                    "model": {
                        **vars(model_args),
                        "resolved_lora_target_modules": sorted(lora_config.target_modules or []),
                    },
                    "data": vars(data_args),
                    "training": training_args.to_dict(),
                },
            },
            kind="sft-training",
            metadata={"model_name_or_path": model_args.model_name_or_path},
        )


if __name__ == "__main__":
    main()
