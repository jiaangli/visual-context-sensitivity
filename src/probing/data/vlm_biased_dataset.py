import io
import logging
import re
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class VLMBiasedDataset:
    def __init__(self, config):
        self.config = config
        self.data = load_dataset(
            config.dataset_name,
            split="main",
            cache_dir=config.dataset_folder,
        )

    def preprocess(self):
        data = self.data.filter(
            lambda x: x["pixel"] == 768 and x["topic"] == "Animals" and self.is_valid_image(x),
            num_proc=self.config.num_workers if self.config.num_workers > 0 else None,
        )

        data = data.map(lambda x: {"prompt": self.format_questions(x, self.config.prompt_format)})
        logger.info(f"Prompt Example: {data['prompt'][0]}")
        mm_inputs = {"image": data["image"], "text": data["prompt"], "ground_truth": data["ground_truth"]}
        return mm_inputs

    def format_questions(self, data_row, prompt_format):
        # adding "in this image" to the question
        # curr_text = text.split("?", 1) if "?" in text else text.split(".", 1)
        # text_body = curr_text[0] + " " + "in THIS image" + "?" if "?" in text else curr_text[0] + "."
        # output_requirement = curr_text[1]
        text = data_row["prompt"]
        image_name = " ".join(re.findall(r"[A-Za-z]+", Path(data_row["image_path"]).stem)).lower()
        curr_text = text.split("?", 1) if "?" in text else text.split(".", 1)
        text_body = (curr_text[0] + "?") if "?" in text else (curr_text[0] + ".")
        if "animal" in text_body.lower():
            text_body = text_body.replace("this animal", "the " + image_name)
        elif "flag" in text_body.lower():
            text_body = text_body.replace("this flag", "the " + image_name)
        elif "shoe" in text_body.lower():
            text_body = text_body.replace("shoe", image_name)
        text_body = prompt_format + text_body
        output_requirement = " Please reason step by step. At the last line, put your final" + curr_text[1].lower()
        # output_requirement = curr_text[1]
        formated_text = text_body + output_requirement

        return formated_text

    @staticmethod
    def is_valid_image(example):
        try:
            if isinstance(example["image"], bytes):
                img = Image.open(io.BytesIO(example["image"]))
            else:
                img = example["image"]
            img.load()  # Will raise an exception if image is corrupted
            return True
        except Exception:
            return False
