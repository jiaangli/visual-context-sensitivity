"""
Data processing utilities for VLind context-sensitivity fine-tuning.

Converts dataset/vlind.jsonl into Qwen2.5-VL chat format.
Each raw entry produces two training samples:
  - context sample: system prompt asks to use the image, label = context_answer
  - prior sample:   system prompt asks to ignore the image, label = prior_answer
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoProcessor

SYSTEM_PROMPT = "Always answer the question in one concise sentence. When the input contains an instruction, follow the instruction exactly."
CONTEXT_INSTRUCTION = "Consider only the {context} to answer the question."
PRIOR_INSTRUCTION = "Ignore the {context} to answer the question."

DEFAULT_VALIDATION_DATASET = "dataset/whatifvis/spatial_temporal_val.jsonl"
DEFAULT_VALIDATION_DATASETS = "dataset/whatifvis/spatial_temporal_val.jsonl,dataset/whatifvis/color_val.jsonl,dataset/whatifvis/size_val.jsonl,dataset/whatifvis/weight_val.jsonl,dataset/whatifvis/count_val.jsonl"
DEFAULT_TEST_DATASETS = "dataset/whatifvis/spatial_temporal_test.jsonl,dataset/whatifvis/color_test.jsonl,dataset/whatifvis/size_test.jsonl,dataset/whatifvis/weight_test.jsonl,dataset/whatifvis/count_test.jsonl"

IGNORE_INDEX = -100
SUFFIX_TEMPLATE_TOKENS = 2
_SEQUENCE_PAD_VALUES = {
    "input_ids": None,
    "attention_mask": 0,
    "labels": IGNORE_INDEX,
    "mm_token_type_ids": 0,
    "_answer_token_mask": 0,
    "_stop_token_mask": 0,
}
_IMAGE_KEYS = {"pixel_values", "image_grid_thw", "image_position_ids"}
SEQUENCE_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids"}


@dataclass
class DataEntry:
    question: str
    context_answer: str
    prior_answer: str
    response_template: str
    statement: Optional[str] = None
    image: Optional[str] = None
    example_id: Optional[str] = None


@dataclass
class SampleMetadata:
    context_answer: str
    prior_answer: str
    mode: str
    question: str
    dataset_name: str
    response_prefix: str
    statement: Optional[str] = None
    image_path: Optional[str] = None
    example_id: Optional[str] = None


@dataclass
class Sample:
    messages: List[dict]
    metadata: SampleMetadata

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for SFTTrainer format detection."""
        return hasattr(self, key)


_img_transform = v2.Compose(
    [
        v2.Resize((512, 512)),
        v2.CenterCrop(512),
    ]
)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def load_jsonl(path: str) -> List[DataEntry]:
    """Load all entries from *.jsonl."""
    entries: List[DataEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            local_example_id = obj.get("example_id") or line_number
            entries.append(
                DataEntry(
                    question=obj["question"],
                    image=obj.get("image"),
                    context_answer=obj["context_answer"],
                    prior_answer=obj["prior_answer"],
                    response_template=obj.get("response_template", "{}"),
                    statement=obj.get("statement"),
                    example_id=f"{Path(path).stem}:{local_example_id}",
                )
            )
    return entries


def resolve_jsonl_image_root(
    jsonl_path: str | Path,
    entries: List[DataEntry],
    image_root: Optional[str],
) -> Optional[str]:
    """Prefer the configured root, then the JSONL directory for relative images."""

    configured_root = Path(image_root) if image_root is not None else None
    local_root = Path(jsonl_path).parent
    for entry in entries:
        if not entry.image or Path(entry.image).is_absolute():
            continue
        normalized_image = entry.image.lower().replace(" ", "_")
        if configured_root is not None and (configured_root / normalized_image).is_file():
            return str(configured_root)
        if (local_root / normalized_image).is_file():
            return str(local_root)
    return image_root


def _format_answer(template: str, answer: str) -> str:
    """Fill response template with the answer word."""
    return template.format(answer) + "."


def build_sample(
    entry: DataEntry,
    mode: str,
    dataset_name: str,
    image_root: Optional[str] = None,
    training: bool = False,
    use_image: bool = True,
    use_instruction: bool = True,
) -> Sample:
    """Build a single sample with messages and metadata."""

    img = None
    if use_image:
        context_type = "image"
        if entry.image is not None:
            image_path = entry.image
            if image_root is not None:
                image_path = str(Path(image_root) / entry.image.lower().replace(" ", "_"))
            img = _img_transform(Image.open(image_path).convert("RGB"))
    else:
        context_type = "statement"

    response_prefix = entry.response_template.split("{}", 1)[0].strip(" ")
    if len(response_prefix) > 0:
        entry.context_answer = " " + entry.context_answer.strip()
        entry.prior_answer = " " + entry.prior_answer.strip()

    if mode == "context":
        answer = entry.context_answer
        instruction = CONTEXT_INSTRUCTION.format(context=context_type)
    else:
        answer = entry.prior_answer
        instruction = PRIOR_INSTRUCTION.format(context=context_type)

    instruct = f"Instruction: {instruction}\n" if use_instruction else ""
    user_content = []

    if use_image:
        user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": f"{instruct}Question: {entry.question}"})
    else:
        user_content.append(
            {
                "type": "text",
                "text": f"Statement: {entry.statement}\n{instruct}Question: {entry.question}",
            }
        )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    if training:
        assistant_content = []
        assistant_content.append({"type": "text", "text": response_prefix + answer + "."})
        messages.append({"role": "assistant", "content": assistant_content})
    else:
        assistant_content = []
        assistant_content.append({"type": "text", "text": response_prefix})
        messages.append({"role": "assistant", "content": assistant_content})

    metadata = SampleMetadata(
        context_answer=entry.context_answer,
        prior_answer=entry.prior_answer,
        mode=mode,
        image_path=entry.image,
        statement=entry.statement,
        question=entry.question,
        dataset_name=dataset_name,
        response_prefix=response_prefix.strip(),
        example_id=entry.example_id,
    )

    return Sample(messages=messages, metadata=metadata)


def _find_subseq_start(subseq: List[int], sequence: torch.Tensor, start: int = 0) -> int:
    """Return the index of the first token of the first occurrence of subseq in sequence."""
    n = len(subseq)
    seq_len = sequence.size(0)
    if n == 0:
        return start  # empty subsequence matches at the start
    if seq_len - start < n:
        return -1  # subsequence is longer than the remaining sequence
    subseq_tensor = sequence.new_tensor(subseq)
    windows = sequence[start:].unfold(0, n, 1)  # shape (seq_len - start - n + 1, n)
    matches = (windows == subseq_tensor).all(dim=1)
    nonzero = matches.nonzero(as_tuple=True)[0]
    if len(nonzero) == 0:
        return -1
    return start + nonzero[0].item() + n


def _mask_labels(input_ids, processor, response_prefix):
    model_name = processor.tokenizer.name_or_path.lower()
    labels = input_ids.clone()
    assistant_prefix = "<|im_start|>assistant\n" if "qwen" in model_name else "<|turn>model\n"

    if "qwen3.5" in model_name:
        response_prefix = "<think>\n\n</think>\n\n" + response_prefix
    assistant_start_tokens = processor.tokenizer.encode(assistant_prefix + response_prefix, add_special_tokens=False)
    asst_start = _find_subseq_start(assistant_start_tokens, input_ids)
    if asst_start == -1:
        raise RuntimeError("Could not find assistant turn start in tokenised sequence.")
    labels[:asst_start] = IGNORE_INDEX
    return labels, asst_start


def _add_answer_stop_masks(data, sample, processor, answer_start):
    """Mark answer and trailing stop/format tokens in one training sequence."""

    answer = sample.metadata.context_answer if sample.metadata.mode == "context" else sample.metadata.prior_answer
    answer_token_ids = processor.tokenizer.encode(
        answer,
        add_special_tokens=False,
    )
    if not answer_token_ids:
        raise RuntimeError("The supervised answer tokenized to an empty sequence")

    answer_end = answer_start + len(answer_token_ids)
    observed_answer_ids = data["input_ids"][answer_start:answer_end].tolist()
    if observed_answer_ids != answer_token_ids:
        raise RuntimeError("Could not align the supervised answer span with the tokenized sequence")

    supervised_mask = data["labels"].ne(IGNORE_INDEX)
    answer_mask = torch.zeros_like(supervised_mask, dtype=torch.bool)
    answer_mask[answer_start:answer_end] = True
    stop_mask = supervised_mask & ~answer_mask
    if not bool(stop_mask.any()):
        raise RuntimeError("The supervised response contains no stop/format tokens")

    data["_answer_token_mask"] = answer_mask
    data["_stop_token_mask"] = stop_mask


def preprocess(sample, processor, training, max_length, normalize_stop_loss_by_token_count: bool = False):
    model_name = processor.tokenizer.name_or_path.lower()
    template_kwargs = dict(
        processor_kwargs={"truncation": True, "max_length": max_length, "padding": False},
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    if "qwen2.5" not in model_name:
        template_kwargs["enable_thinking"] = False

    encoding = processor.apply_chat_template(sample.messages, **template_kwargs)
    data = {}
    for k, v in encoding.items():
        if k in SEQUENCE_KEYS and torch.is_tensor(v) and v.dim() == 2:
            data[k] = v[0]
        else:
            data[k] = v

    if training:
        labels, answer_start = _mask_labels(data["input_ids"], processor, sample.metadata.response_prefix)
        data["labels"] = labels
        if normalize_stop_loss_by_token_count:
            _add_answer_stop_masks(data, sample, processor, answer_start)
    else:
        for key in SEQUENCE_KEYS:
            if key in data:
                data[key] = data[key][:-SUFFIX_TEMPLATE_TOKENS]
        data["metadata"] = sample.metadata
    return data


# def _pad_fixed(input, padding_value, target_len, padding_side):
#     out = torch.nn.utils.rnn.pad_sequence(
#         input, batch_first=True, padding_value=padding_value, padding_side=padding_side
#     )
#     gap = target_len - out.size(1)
#     if gap > 0:
#         cfg = (0, gap) if padding_side == "right" else (gap, 0)
#         out = F.pad(out, cfg, value=padding_value)
#     return out


class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
    Loads and builds samples lazily to minimize memory usage.

    Args:
        jsonl_path: Path to the dataset jsonl file
        image_root: Optional root directory for image paths
        start: Start index for slicing
        stop: Stop index for slicing (None = end of dataset)
        step: Step size for slicing
    """

    def __init__(
        self,
        jsonl_path: str,
        processor,
        image_root: Optional[str] = None,
        training: bool = True,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
        use_image: bool = True,
        use_instruction: bool = True,
        normalize_stop_loss_by_token_count: bool = False,
        max_length: int = 512,
        shuffle: bool = False,
        **kwargs,
    ):
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be >= start, got {stop} and {start}"
        assert step >= 1, f"Step must be >= 1, got {step}"

        self.entries = load_jsonl(jsonl_path)
        if shuffle:
            random.shuffle(self.entries)
        self.entries = self.entries
        self.dataset_name = Path(jsonl_path).stem
        self.image_root = resolve_jsonl_image_root(jsonl_path, self.entries, image_root)
        self.training = training
        self.use_image = use_image
        self.use_instruction = use_instruction
        self.normalize_stop_loss_by_token_count = normalize_stop_loss_by_token_count
        self.start = start
        self.stop = stop
        self.step = step
        self.processor = processor
        self.max_length = max_length

    def num_examples(self) -> int:
        return len(self.entries) * 2

    def __len__(self) -> int:
        start = self.start
        stop = min(self.num_examples(), self.stop) if self.stop is not None else self.num_examples()
        span = stop - start
        return ceil_div(span, self.step)

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step

        # Even index is context sample, odd is prior sample
        entry_idx = physical_index // 2
        mode = "context" if physical_index % 2 == 0 else "prior"

        entry = self.entries[entry_idx]
        sample = build_sample(
            entry,
            mode,
            self.dataset_name,
            training=self.training,
            image_root=self.image_root,
            use_image=self.use_image,
            use_instruction=self.use_instruction,
        )
        item = preprocess(
            sample,
            self.processor,
            self.training,
            self.max_length,
            normalize_stop_loss_by_token_count=self.normalize_stop_loss_by_token_count,
        )
        return item


class TaskMixture(Task):
    """
    Mixture of multiple tasks with optional shuffling.
    Loads samples lazily from the underlying tasks.

    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.

    Args:
        tasks: List of Task objects
        shuffle: Whether to shuffle the task mixture (uses the global random state)
        start: Start index for slicing
        stop: Stop index for slicing (None = end of dataset)
        step: Step size for slicing
    """

    def __init__(
        self,
        tasks: List[Task],
        shuffle: bool = True,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
    ):
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be >= start, got {stop} and {start}"
        assert step >= 1, f"Step must be >= 1, got {step}"

        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

        self.index_map: List[tuple] = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))

        if shuffle:
            random.shuffle(self.index_map)

        self.start = start
        self.stop = stop
        self.step = step

    def num_examples(self) -> int:
        return self.num_conversations

    def __len__(self) -> int:
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        span = stop - start
        return ceil_div(span, self.step)

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        task_idx, local_idx = self.index_map[physical_index]
        return self.tasks[task_idx][local_idx]


class CustomCollator:
    """
    Collates a batch of Sample objects into model inputs with masked labels.

    The processor applies the Qwen2.5-VL chat template, tokenises the text, and
    processes images. Labels are set to IGNORE_INDEX for:
      - All user/image prompt tokens (everything before the assistant turn).
      - The response template prefix tokens (e.g. "The deserts are " in
        "The deserts are sandy"), so loss is only computed on the answer word(s).

    Args:
        processor: HuggingFace processor for the model
        max_length: Maximum sequence length
        inference_mode: If True, prepares inputs for inference (generation_prompt=True, no label masking)
    """

    def __init__(
        self,
        processor: AutoProcessor,
        max_length: int = 512,
        inference_mode: bool = False,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.inference_mode = inference_mode

    def __call__(self, batch: List) -> Dict[str, Any]:
        """
        Args:
            batch: list of Sample objects (from Task.__getitem__)

        Returns:
            dict with keys: input_ids, attention_mask, pixel_values (optional),
                            image_grid_thw (optional), mm_token_type_ids (optional),
                            labels (training only)
        """
        tokenizer = self.processor.tokenizer
        padding_side = tokenizer.padding_side
        first = batch[0]
        encoding = {}
        handled = {"metadata"}

        for key, pad_value in _SEQUENCE_PAD_VALUES.items():
            if key not in first:
                continue
            if pad_value is None:
                pad_value = tokenizer.pad_token_id
            encoding[key] = torch.nn.utils.rnn.pad_sequence(
                [_sample[key] for _sample in batch],
                batch_first=True,
                padding_value=pad_value,
                padding_side=padding_side,
            )
            handled.add(key)

        for key in first:
            if key in handled:
                continue
            values = [_sample[key] for _sample in batch]
            encoding[key] = torch.cat(values, dim=0) if torch.is_tensor(values[0]) else values

        if self.inference_mode:
            encoding["_metadata"] = [_sample["metadata"] for _sample in batch]
        return encoding
