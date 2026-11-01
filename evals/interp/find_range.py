import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    HfArgumentParser,
)

from src.eval_results import (
    DEFAULT_EVAL_ROOT,
    configure_run_logging,
    prepare_eval_run,
    write_json_result,
)
from src.interp.api.gemma4 import Gemma4
from src.interp.api.qwen2p5vl import Qwen2p5VL
from src.interp.api.qwen3p5 import Qwen3p5
from src.interp.das import auto_search_full_answer
from src.interp.data import (
    PatchingData,
    filter_by_full_answer_preferences,
    prepare_vlm_patching_data,
    transform_patching_data,
)
from src.interp.decoding import get_num_layers
from src.interp.layer_selection import (
    DEFAULT_ONSET_PERSISTENCE,
    DEFAULT_ONSET_THRESHOLD,
)
from src.model_utils import setup_seed
from src.sft.data_process import DEFAULT_VALIDATION_DATASET
from src.sft.preference_eval import candidate_token_ids

load_dotenv()
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


@dataclass
class AnalysisArguments:
    expid: str = field(
        default="activation-patching-range",
        metadata={"help": "Experiment id used to group related layer-search jobs"},
    )
    eval_category: str = field(default="find_layers", metadata={"help": "Experiment result category"})
    output_root: str = field(default=DEFAULT_EVAL_ROOT, metadata={"help": "Root directory for eval artifacts"})
    timestamp: Optional[str] = field(
        default=None,
        metadata={"help": "Shared UTC run timestamp (YYYYMMDDThhmmssZ); generated when omitted"},
    )
    output_name: Optional[str] = field(
        default=None,
        metadata={"help": "Result filename inside the expid/timestamp run directory"},
    )
    task_direction: str = field(
        default="prior_to_context",
        metadata={"help": "Directions 'prior_to_context', 'context_to_prior', 'prior_only', 'context_only'"},
    )
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Base model path"},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to LoRA adapter (if any)"},
    )
    dataset_path: str = field(
        default=DEFAULT_VALIDATION_DATASET,
        metadata={"help": "Validation JSONL used to select the layer range"},
    )
    image_root: Optional[str] = field(
        default="dataset/whatifvis",
        metadata={"help": "Root directory for images"},
    )
    batch_size: int = field(default=32, metadata={"help": "Batch size for analysis"})
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum samples to use (None = all)"},
    )
    search_objective: str = field(
        default="full_answer",
        metadata={"help": "Search objective; only 'full_answer' is supported"},
    )
    search_retention_threshold: float = field(
        default=0.8,
        metadata={"help": "Minimum baseline-normalized full-answer effect R(S)"},
    )
    search_onset_threshold: float = field(
        default=DEFAULT_ONSET_THRESHOLD,
        metadata={"help": ("Minimum probability-gap effect used to detect the recommended single-layer onset")},
    )
    search_onset_persistence: int = field(
        default=DEFAULT_ONSET_PERSISTENCE,
        metadata={"help": "Consecutive nondecreasing layers required above the onset threshold"},
    )
    search_max_length: int = field(
        default=512,
        metadata={"help": "Maximum prompt-plus-answer length for full-answer scoring"},
    )
    seed: int = field(default=42, metadata={"help": "Global random seed, set once at startup"})
    skip_filter: bool = field(
        default=False,
        metadata={"help": "Skip correctness filtering; use all loaded samples directly"},
    )
    require_distinct_answer_tokens: bool = field(
        default=True,
        metadata={"help": "Require prior/context gold answers to have different first tokens"},
    )
    full_attention_only: bool = field(
        default=False,
        metadata={"help": "Search only layers whose configured type is full_attention"},
    )
    skip_layer_search: bool = field(
        default=False,
        metadata={"help": "Skip layer search, use provided layer"},
    )
    target_layer: Optional[int] = field(
        default=None,
        metadata={"help": "Target layer for intervention (if skipping search)"},
    )
    use_image: bool = field(
        default=True,
        metadata={"help": "Whether to include images in the analysis"},
    )
    cache_dir: Optional[str] = field(
        default="./cache",
        metadata={"help": "Cache directory for filtering results"},
    )
    use_cache: bool = field(
        default=False,
        metadata={"help": "Whether to use cache when doing the filtering"},
    )


def build_filter_cache_path(args):
    model_tag = Path(args.model_name_or_path).name
    if args.lora_path is not None:
        model_tag = f"{Path(args.lora_path).name}"
    dataset_tag = Path(args.dataset_path).stem
    image_tag = "with_image" if args.use_image else "wo_image"

    if args.skip_filter:
        filter_tag = "distinct_only" if args.require_distinct_answer_tokens else "unfiltered"
    else:
        filter_tag = (
            "full_answer_both_preferred_distinct"
            if args.require_distinct_answer_tokens
            else "full_answer_both_preferred"
        )
    identity = "\0".join(
        (
            args.model_name_or_path,
            args.lora_path or "",
            str(Path(args.dataset_path).expanduser().resolve()),
            args.task_direction,
            image_tag,
            str(args.max_samples),
            str(args.seed),
            args.search_objective,
            filter_tag,
            # Data order comes from the global RNG stream; don't reuse caches built under per-call seeding.
            "global_rng",
        )
    )
    cache_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    tag = "_".join(
        [
            model_tag,
            args.task_direction,
            image_tag,
            f"max{args.max_samples}" if args.max_samples is not None else "all",
            dataset_tag,
            f"seed{args.seed}",
            filter_tag,
            cache_digest,
        ]
    )
    tag = re.sub(r"[^A-Za-z0-9_.=-]", "_", tag)
    cache_path = Path(args.cache_dir) / f"patching_data_{tag}.pt"
    return cache_path


def _answer_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_full_answers.json")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _full_answer_ids_for_direction(
    processor,
    context_metadatas,
    prior_metadatas,
    task_direction: str,
    source_indices,
    target_indices,
):
    context_ids = candidate_token_ids(
        processor,
        [metadata.context_answer for metadata in context_metadatas],
        [metadata.response_prefix for metadata in context_metadatas],
    )
    prior_ids = candidate_token_ids(
        processor,
        [metadata.prior_answer for metadata in prior_metadatas],
        [metadata.response_prefix for metadata in prior_metadatas],
    )
    if task_direction == "prior_to_context":
        source_pool, target_pool = prior_ids, context_ids
    elif task_direction == "context_to_prior":
        source_pool, target_pool = context_ids, prior_ids
    elif task_direction == "prior_only":
        source_pool = target_pool = prior_ids
    elif task_direction == "context_only":
        source_pool = target_pool = context_ids
    else:
        raise ValueError(f"Unknown task_direction: {task_direction}")
    return (
        [list(source_pool[index]) for index in source_indices],
        [list(target_pool[index]) for index in target_indices],
    )


def _slice_patching_data(data: PatchingData, max_samples: int) -> PatchingData:
    fields = {}
    for name in data.__dataclass_fields__:
        value = getattr(data, name)
        fields[name] = None if value is None else value[:max_samples]
    return PatchingData(**fields)


def _select_patching_data_rows(data: PatchingData, indices) -> PatchingData:
    selected = {}
    for name in data.__dataclass_fields__:
        value = getattr(data, name)
        if value is None:
            selected[name] = None
        else:
            index = torch.tensor(indices, dtype=torch.long, device=value.device)
            selected[name] = value[index]
    return PatchingData(**selected)


def _write_search_trace(path: Path, search_trace: dict) -> None:
    """Persist replayable plot data before the final result is written"""
    _write_json_atomic(path, search_trace)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = get_device()


def get_model_api(model_name_or_path):
    name = model_name_or_path.lower()
    if "qwen2.5" in name:
        return Qwen2p5VL
    elif "qwen3.5" in name:
        return Qwen3p5
    elif "gemma" in name:
        return Gemma4
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")


def get_searchable_layer_indices(model):
    text_config = model.config.text_config
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is None:
        return None
    full_attn = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    if len(full_attn) == len(layer_types):
        return None
    return full_attn


def load_model_and_processor(args: AnalysisArguments, device: str):
    logger.info(f"Loading model from: {args.model_name_or_path}")

    if args.lora_path is not None:
        logger.info(f"Loading LoRA adapter from: {args.lora_path}")
        base_model = AutoModelForMultimodalLM.from_pretrained(
            args.model_name_or_path,
            dtype=torch.bfloat16,
            device_map="auto",
            return_dict=True,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained(args.lora_path, use_fast=True)
    else:
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model_name_or_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, use_fast=True)

    model.config.use_cache = False
    model.eval()
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.chat_template = "{%- set enable_thinking = false %}\n" + processor.tokenizer.chat_template

    return model, processor


def main():
    parser = HfArgumentParser((AnalysisArguments,))
    (args,) = parser.parse_args_into_dataclasses()

    valid_directions = {
        "prior_to_context",
        "context_to_prior",
        "prior_only",
        "context_only",
    }
    if args.task_direction not in valid_directions:
        raise ValueError(
            f"Unsupported task_direction {args.task_direction!r}; choose one of {sorted(valid_directions)}"
        )
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {args.max_samples}")
    if args.search_objective != "full_answer":
        raise ValueError(f"search_objective must be 'full_answer', got {args.search_objective!r}")
    if not 0 < args.search_retention_threshold <= 1:
        raise ValueError(f"search_retention_threshold must be in (0, 1], got {args.search_retention_threshold}")
    if not 0 < args.search_onset_threshold <= 2:
        raise ValueError(f"search_onset_threshold must be in (0, 2], got {args.search_onset_threshold}")
    if args.search_onset_persistence <= 0:
        raise ValueError(f"search_onset_persistence must be positive, got {args.search_onset_persistence}")
    if args.search_max_length <= 0:
        raise ValueError(f"search_max_length must be positive, got {args.search_max_length}")
    if args.use_cache and args.cache_dir is None:
        raise ValueError("cache_dir is required when use_cache=True")

    model_tag = Path(args.model_name_or_path).name
    modality = "image" if args.use_image else "text"
    output_name = args.output_name or f"{model_tag}_{args.task_direction}_{modality}_layer_range.json"
    run = prepare_eval_run(
        category=args.eval_category,
        expid=args.expid,
        output_root=args.output_root,
        timestamp=args.timestamp,
        output_name=output_name,
    )
    configure_run_logging(run, kind="activation-patching-layer-search")
    setup_seed(args.seed)

    device = get_device()
    logger.info(f"Using device: {device}")

    model, processor = load_model_and_processor(args, device)

    model_api = get_model_api(args.model_name_or_path)
    logger.info(f"Using model API: {model_api.__name__}")

    n_layers = get_num_layers(model)
    logger.info(f"Model has {n_layers} layers")

    layer_indices = get_searchable_layer_indices(model) if args.full_attention_only else None
    if layer_indices is not None:
        logger.info(
            f"Model has {len(layer_indices)} searchable layers (full attention layers) at indices: {layer_indices}"
        )

    logger.info(f"Loading dataset from: {args.dataset_path}")

    cache_path = build_filter_cache_path(args) if args.use_cache else None
    answer_cache_path = _answer_cache_path(cache_path) if cache_path is not None else None
    cache_ready = cache_path is not None and cache_path.exists() and answer_cache_path.exists()

    filter_stats = None
    source_answer_ids = None
    target_answer_ids = None
    if cache_ready:
        logger.info(f"Loading cached filtered data from: {cache_path}")
        data = PatchingData.load(str(cache_path))
        answer_payload = json.loads(answer_cache_path.read_text(encoding="utf-8"))
        source_answer_ids = answer_payload["source_answer_ids"]
        target_answer_ids = answer_payload["target_answer_ids"]
        filter_stats = answer_payload.get("filter_stats")
    else:
        # Cache miss: prepare -> transform -> filter -> save
        context_metadatas, prior_metadatas, data = prepare_vlm_patching_data(
            processor=processor,
            dataset_path=args.dataset_path,
            image_root=args.image_root,
            device=device,
            use_image=args.use_image,
        )
        logger.info(f"Loaded {data.source_input_ids.shape[0]} samples")
        logger.info(f"Use images: {args.use_image}")
        direction_labels = {
            "prior_to_context": "prior -> context",
            "context_to_prior": "context -> prior",
            "prior_only": "prior only",
            "context_only": "context only",
        }
        logger.info(f"Analyzing {direction_labels[args.task_direction]}")
        data, source_indices, target_indices = transform_patching_data(
            data,
            context_metadatas,
            prior_metadatas,
            args.task_direction,
            return_pair_indices=True,
        )

        source_answer_ids, target_answer_ids = _full_answer_ids_for_direction(
            processor,
            context_metadatas,
            prior_metadatas,
            args.task_direction,
            source_indices,
            target_indices,
        )

        if not args.skip_filter:
            logger.info("Filtering with complete-answer preferences on both prompts...")
            data, source_answer_ids, target_answer_ids, filter_stats = filter_by_full_answer_preferences(
                model=model,
                processor=processor,
                data=data,
                source_answer_ids=source_answer_ids,
                target_answer_ids=target_answer_ids,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                max_length=args.search_max_length,
                normalize_to_default_size=False,
                require_distinct_answer_tokens=args.require_distinct_answer_tokens,
                return_filter_stats=True,
            )
        else:
            if args.require_distinct_answer_tokens:
                total_pairs = len(source_answer_ids)
                distinct_indices = [
                    index
                    for index, (source_ids, target_ids) in enumerate(zip(source_answer_ids, target_answer_ids))
                    if source_ids[0] != target_ids[0]
                ]
                data = _select_patching_data_rows(data, distinct_indices)
                source_answer_ids = [source_answer_ids[i] for i in distinct_indices]
                target_answer_ids = [target_answer_ids[i] for i in distinct_indices]
                filter_stats = {
                    "criterion": "different_first_token_only",
                    "total_training_pairs": total_pairs,
                    "different_first_token_pairs": len(distinct_indices),
                }

            if args.max_samples is not None:
                logger.info("Subsampling to max samples...")
                data = _slice_patching_data(data, args.max_samples)
                source_answer_ids = source_answer_ids[: args.max_samples]
                target_answer_ids = target_answer_ids[: args.max_samples]
            logger.info("Skipping correctness filtering (skip_filter)")

        if cache_path is not None:
            data.save(str(cache_path))
            _write_json_atomic(
                answer_cache_path,
                {
                    "source_answer_ids": source_answer_ids,
                    "target_answer_ids": target_answer_ids,
                    "filter_stats": filter_stats,
                },
            )

    if data.source_input_ids.shape[0] == 0:
        raise ValueError("No input pairs remain after filtering/subsampling")

    # Expand the data set by interleaving "Ignore Context" and "UseContext" samples from the source data, while ensuring that the target data
    # data = expand_patching_data(data, indices=None)

    layer_range = None
    target_layer = args.target_layer if args.skip_layer_search else None
    search_trace = None
    search_trace_path = None
    if args.skip_layer_search:
        if args.target_layer is None:
            args.target_layer = n_layers // 2
            logger.info(f"No target layer specified, using middle layer: {args.target_layer}")
        target_layer = args.target_layer
    else:
        logger.info(f"Running %s layer search...", args.search_objective)
        search_trace = {}
        search_trace_path = run.artifact_path("search_trace/trace.json")

        def persist_search_trace(trace):
            trace["task_direction"] = args.task_direction
            trace["n_samples"] = int(data.source_input_ids.shape[0])
            trace["run_context"] = {
                "model_name_or_path": args.model_name_or_path,
                "lora_path": args.lora_path,
                "dataset_path": args.dataset_path,
                "image_root": args.image_root,
                "use_image": args.use_image,
                "seed": args.seed,
                "search_objective": args.search_objective,
                "filter_stats": filter_stats,
            }
            trace["artifacts"] = {"trace_json": str(search_trace_path)}
            _write_search_trace(search_trace_path, trace)

        try:
            with torch.inference_mode():
                layer_range = auto_search_full_answer(
                    model=model,
                    processor=processor,
                    patching_arguments=data,
                    source_answer_ids=source_answer_ids,
                    target_answer_ids=target_answer_ids,
                    n_layers=n_layers,
                    retention_threshold=args.search_retention_threshold,
                    batch_size=args.batch_size,
                    api=model_api,
                    max_samples=args.max_samples,
                    max_length=args.search_max_length,
                    layer_indices=layer_indices,
                    search_trace=search_trace,
                    trace_callback=persist_search_trace,
                    collect_plot_data=True,
                    onset_threshold=args.search_onset_threshold,
                    onset_persistence=args.search_onset_persistence,
                )
        except Exception as error:
            search_trace["status"] = "failed"
            search_trace.setdefault("error", f"{type(error).__name__}: {error}")
            persist_search_trace(search_trace)
            logger.exception("Layer search failed; partial trace was saved")
            raise
        else:
            logger.info(f"Found important layers: {layer_range}")
            if search_trace.get("target_layer") is not None:
                target_layer = int(search_trace["target_layer"])
                logger.info("Selected difference-onset target layer: %d", target_layer)
            logger.info("=" * 25 + "Finished layer search" + "=" * 25)
            persist_search_trace(search_trace)
        logger.info("Saved reusable search trace to %s", search_trace_path)

    write_json_result(
        run,
        {
            "config": asdict(args),
            "important_layers": layer_range,
            "target_layer": target_layer,
            "n_samples": int(data.source_input_ids.shape[0]),
            "filter_stats": filter_stats,
            "search_trace": search_trace,
            "search_trace_file": str(search_trace_path) if search_trace_path is not None else None,
        },
        kind="activation-patching-layer-search",
    )
    logger.info("Saved layer-search result to %s", run.result_path)


if __name__ == "__main__":
    main()
