import io
import logging
import random
from functools import partial
from pathlib import Path

import numpy as np
import PIL
import torch
import webdataset as wds
from datasets import Image, load_dataset
from torchvision.transforms import v2

logger = logging.getLogger(__name__)

# ImageNet training shards under `training_args.data_dir`. Precomputed latents
# (app/imagegen/preprocess.py) take precedence over raw-image shards.
IMAGENET_LATENT_SUBDIRS = ("imagenet-1k-latent", "imagenet-1k-latent-512")
IMAGENET_IMAGE_SUBDIRS = ("imagenet-1k-wds",)
IMAGENET_SHARD_GLOB = "imagenet1k-train-*.tar"


def _list_shards(data_dir, subdirs):
    return [str(path) for subdir in subdirs for path in sorted(Path(data_dir, subdir).glob(IMAGENET_SHARD_GLOB))]


def resolve_imagenet_shards(data_dir):
    """Choose the ImageNet source once for the whole run.

    Returns ``(shard_urls, use_latents)``: latent shards are read as
    precomputed VAE latents, raw-image shards are VAE-encoded on the fly.
    """
    latent_urls = _list_shards(data_dir, IMAGENET_LATENT_SUBDIRS)
    if latent_urls:
        return latent_urls, True
    image_urls = _list_shards(data_dir, IMAGENET_IMAGE_SUBDIRS)
    if image_urls:
        return image_urls, False
    searched = ", ".join(str(Path(data_dir, d)) for d in IMAGENET_LATENT_SUBDIRS + IMAGENET_IMAGE_SUBDIRS)
    raise FileNotFoundError(f"No ImageNet shards matching {IMAGENET_SHARD_GLOB} in: {searched}")


class BlankImageReplacer:
    """Replaces a fraction of training images with a black square.

    Small, non-RGB or extreme-aspect-ratio images are always blanked; the rest
    are blanked at random so the overall blank fraction tracks `blank_ratio`.
    Shared by latent precomputation (app/imagegen/preprocess.py) and raw-image
    training so both ImageNet sources get the same augmentation. Counters are
    per instance, i.e. per DataLoader worker; draws use the global `random`.
    """

    def __init__(self, blank_ratio=0.1, blank_size=512, min_image_size=224, max_aspect_ratio=2.0):
        self.blank_ratio = blank_ratio
        self.min_image_size = min_image_size
        self.max_aspect_ratio = max_aspect_ratio
        self.blank_image = PIL.Image.new("RGB", (blank_size, blank_size), color=(0, 0, 0))
        self.seen = 0
        self.blanked = 0

    def is_small_or_weird(self, img):
        width, height = img.size
        return (
            min(width, height) <= self.min_image_size
            or img.mode != "RGB"
            or max(width / height, height / width) >= self.max_aspect_ratio
        )

    def __call__(self, img):
        """Returns ``(image, blanked)``; ``image`` is the shared blank image when blanked."""
        if self.blank_ratio <= 0:
            return img, False
        self.seen += 1
        if self.is_small_or_weird(img):
            blank = True
        else:
            budget = self.blank_ratio * self.seen - self.blanked
            blank = budget > 0 and random.random() < min(1.0, budget)
        if blank:
            self.blanked += 1
            return self.blank_image, True
        return img, False


def decode_and_preprocess(sample, target_transform, use_latents, blank_replacer=None):
    """Decodes one shard sample into the model inputs and the diffusion target.

    Latent shards already carry the blank augmentation (applied when the
    latents were computed), so `blank_replacer` is only used for raw images.
    Errors propagate to the WebDataset handler, which warns and skips the sample.
    """
    img = PIL.Image.open(io.BytesIO(sample["jpg"]))
    if use_latents:
        latents = torch.from_numpy(np.load(io.BytesIO(sample["latent.npy"])))
        target_img = None
    else:
        if blank_replacer is not None:
            img, _ = blank_replacer(img)
        # Raw ImageNet contains grayscale and CMYK JPEGs; the VAE expects RGB.
        if img.mode != "RGB":
            img = img.convert("RGB")
        latents = None
        target_img = target_transform(img)

    return {
        "target": target_img,
        "caption": "",
        "input_images": img,
        "latents": latents,
    }


def is_truncated_image(img):
    """Checks if the error is due to a truncated image."""
    try:
        img = PIL.Image.open(io.BytesIO(img["bytes"]) if img["bytes"] is not None else img["path"]).convert("RGB")
    except Exception:
        img = None
    return img is not None


class CustomCollator:
    """Collates samples into a batch for the model."""

    def __init__(self, tokenize_fn, tokenizer, model_args):
        self.tokenizer = tokenizer
        self.tokenize_fn = tokenize_fn
        self.model_args = model_args

    def __call__(self, batch):
        captions = [item["caption"] for item in batch]
        input_images = [[item["input_images"]] for item in batch]

        inputs = self.tokenize_fn(self.tokenizer, captions, input_images)

        if inputs.get("pixel_values", None) is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # The source is fixed per run, so a batch is either all latents or all images.
        if batch[0]["latents"] is not None:
            return {"latents": torch.stack([item["latents"] for item in batch]), **inputs}
        # Keyword must match `MetaQuery.forward(target=...)`, which VAE-encodes it.
        return {"target": torch.stack([item["target"] for item in batch]), **inputs}


def get_train_datasets(data_args, training_args, model_args, tokenize_fn, tokenizer):
    """Prepares the training and evaluation datasets using WebDataset."""
    resize_input_size = data_args.target_image_size
    target_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.CenterCrop(data_args.target_image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )
    ground_truth_transform = v2.Compose(
        [
            v2.Resize((data_args.target_image_size, data_args.target_image_size)),
            v2.CenterCrop(data_args.target_image_size),
        ]
    )

    train_input_transform = v2.Compose(
        [
            v2.Resize(resize_input_size),
            v2.CenterCrop(resize_input_size),
        ]
    )

    eval_input_transform = v2.Compose(
        [
            v2.Resize((resize_input_size, resize_input_size)),
            v2.CenterCrop(resize_input_size),
        ]
    )

    train_urls, use_latents = resolve_imagenet_shards(training_args.data_dir)
    logger.info(
        "ImageNet training source: %s (%d shards, first: %s)",
        "precomputed latents" if use_latents else "raw images, VAE-encoded on the fly",
        len(train_urls),
        train_urls[0],
    )
    # Create the training WebDataset pipeline
    train_dataset = (
        wds.WebDataset(train_urls, seed=training_args.seed, resampled=True, nodesplitter=wds.split_by_node)
        .map(
            partial(
                decode_and_preprocess,
                target_transform=target_transform,
                use_latents=use_latents,
                blank_replacer=None
                if use_latents
                else BlankImageReplacer(
                    blank_ratio=data_args.blank_ratio,
                    blank_size=data_args.target_image_size,
                    min_image_size=data_args.blank_min_image_size,
                    max_aspect_ratio=data_args.blank_max_aspect_ratio,
                ),
                # input_transform=train_input_transform,
            ),
            handler=wds.handlers.warn_and_continue,
        )
        .with_epoch(
            training_args.max_steps
            * training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
        )  # type: ignore
    )
    if "biased" in data_args.eval_dataset:
        eval_dataset = load_dataset(
            "anvo25/vlms-are-biased",
            cache_dir=f"{training_args.data_dir}/hf_data",
            split="identification",
            num_proc=training_args.datasets_num_proc,
        )
        eval_dataset = eval_dataset.cast_column("image", Image(decode=False))
        eval_dataset = eval_dataset.filter(
            lambda x: x["pixel"] == 1152 and x["topic"] in ["Animals"] and is_truncated_image(x["image"]),
            num_proc=training_args.datasets_num_proc,  # type: ignore
        )
        eval_dataset = eval_dataset.select(range(16))
        eval_dataset = eval_dataset.cast_column("image", Image(decode=True))
        # Empty captions for ImageNet
        eval_dataset = eval_dataset.add_column("caption", [""] * len(eval_dataset))  # type: ignore
        eval_dataset = eval_dataset.map(
            lambda x: {"input_images": eval_input_transform(x["image"]).convert("RGB")},
            remove_columns=[col for col in eval_dataset.column_names if col not in ["input_images", "caption"]],
            num_proc=training_args.datasets_num_proc,
        )
        gt_images = (
            eval_dataset["target_image"]
            if "target_image" in eval_dataset.column_names
            else eval_dataset["input_images"]
        )
        gt_images = [ground_truth_transform(image.convert("RGB")) for image in gt_images]

    collate_fn = CustomCollator(
        tokenize_fn=tokenize_fn,
        tokenizer=tokenizer,
        model_args=model_args,
    )

    return train_dataset, eval_dataset, gt_images, collate_fn
