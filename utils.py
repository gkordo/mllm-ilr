import torch
import yaml
import argparse

from types import SimpleNamespace
from transformers import AutoProcessor, BatchFeature

binary_prompt = """
You are given two images: a query and a candidate. Determine if they show the same unique instance, not just similar objects. Use fine-grained visual details (e.g., marks, textures, logos) and identity/location-specific cues. The same instance may appear at a different scale, under different brightness, in a different orientation, or partially occluded. Ignore general similarity (e.g., same type or model).

Output only a single number:
0 = different instances
1 = same unique instance

No extra text. Just the number.
"""

class_prompt = """
You are given two images: a query and a candidate. Determine whether the candidate image contains an object that belongs to the same semantic class as the object in the query image.
* The object does not need to be the same instance, only the same class type.
* It may appear at a different scale.
* It may be partially visible or occluded.
* Other objects may also appear in the candidate.

Output strictly a single digit:
0 = an instance of the same class does not appear in the candidate
1 = an instance of the same class appears in the candidate

Do not output anything else.
"""

class_simple_prompt = """
You are given two images: a query and a candidate. Determine whether the candidate belongs to the same semantic class as the query image.

Output strictly a single digit:
0 = the two images are of a different class
1 = the two images are of the same class

Do not output anything else.
"""

object_details_prompt = """
You are given two images: a query and a candidate. Determine whether the exact same object instance from the query image is present in the candidate image.
* The instance must be the same, not just a similar object.
* The instance may appear at a different scale
* It may be partially visible or occluded.
* Other objects may also appear in the candidate.

Output strictly a single digit:
0 = the object instance does not appear
1 = the object instance appears in the candidate

Do not output anything else.
"""

object_prompt = """
You are given two images: a query and a candidate. Determine whether the exact same object instance from the query image is present in the candidate image.
* The instance must be the same, not just a similar object.
* The instance may appear at a different scale, partially occluded, or among other objects.

Output strictly a single digit:
0 = the object instance does not appear
1 = the object instance appears in the candidate

Do not output anything else.
"""

inat_prompt = """
You are given two images: a query and a candidate. Determine whether the exact same fine-grained biological species (e.g., a specific type of animal, bird, insect, or plant) from the query image is present in the candidate image.
* Pay close attention to fine-grained differences. Do not match two different species just because they are visually similar (e.g., a "Monarch Butterfly" and a "Viceroy Butterfly" are a 0).
* The same species may be shown from a different angle, at a different scale, or be partially occluded.

Output strictly a single digit: 
0 = the same species does not appear in the candidate 
1 = the same species appears in the candidate

Do not output anything else.
"""

landmark_prompt = """
You are given two images: a query and a candidate. Determine whether the exact same landmark, building, or architectural detail from the query image is present in the candidate image.
* The instance must be the same, not just a similar-looking building or structure.
* The query image may show the entire landmark or just a specific, cropped part of it (like a doorway, statue, or window).
* The instance in the candidate image may appear at a different scale, from a different viewpoint/angle, under different lighting, or be partially occluded.

Output strictly a single digit:
0 = the instance does not appear
1 = the instance appears in the candidate

Do not output anything else.
"""

met_prompt = """
You are given two images: a query and a candidate. Determine whether the exact same artwork from the query image is present in the candidate image.
* The instance must be the same, not just a similar-looking artwork or one of the same subject.
* The instance may appear at a different scale, from a different viewpoint, partially occluded, or among other objects.

Output strictly a single digit:
0 = the instance does not appear
1 = the instance appears in the candidate

Do not output anything else.
"""

object_yesno_prompt = """
You are given two images: a query and a candidate. Determine whether the exact same object instance from the query image is present in the candidate image.
* The instance must be the same, not just a similar object.
* The instance may appear at a different scale, partially occluded, or among other objects.
Output strictly a yes or no and nothing else:
Yes = the object instance from the query appears in the candidate
No = the object instance from the query does NOT appear in the candidate
Do not output anything else.
"""

object_simple_prompt = """
You are given two images: a query and a candidate. Determine if the exact same object instance from the query image is present in the candidate image (not just a similar object).

Output strictly a single digit:
0 = the object instance does not appear
1 = the object instance appears in the candidate

Do not output anything else.
"""

ternary_prompt = """
You are given two images: a query and a candidate. Determine whether they depict the same unique instance (e.g., the same exact object or individual), not just similar ones.

Focus on fine-grained visual cues (e.g., scratches, wear, textures, logos, location context). Ignore general similarity such as same model, brand, or object type. The same instance may appear at a different scale, under different brightness, in a different orientation, or partially occluded.

Output only a single number, with no additional text:

0 = Clearly different instances.
1 = Possibly the same unique instance, but uncertain.
2 = Definitely the same unique instance.
"""

decimal_prompt = """
You are given two images: a query and a candidate. Determine whether they depict the same unique instance (e.g., the same exact object or individual), not just similar ones.

Focus on fine-grained visual cues (e.g., scratches, wear, textures, logos, location context). Ignore general similarity such as same model, brand, or object type. The same instance may appear at a different scale, under different brightness, in a different orientation, or partially occluded.

Output only a single number between 1 and 10, with no additional text:

1-3: Clearly different instances.
4-6: Same object category or class but clearly different instances.
7-8: Likely the same instance but with some uncertainty.
9: Very likely the same instance.
10: Almost certainly the same unique instance.
"""


def bool_flag(s):
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def resize_to_fit(image_shape, required_dim):
    original_height, original_width = image_shape[:2]
    aspect_ratio = original_width / original_height

    if original_width >= original_height:
        resized_width = required_dim
        resized_height = int(required_dim / aspect_ratio)
    else:
        resized_height = required_dim
        resized_width = int(required_dim * aspect_ratio)

    return resized_height, resized_width


def dict_to_namespace(d):
    if isinstance(d, dict):
        ns = SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, dict_to_namespace(v))
        return ns
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return dict_to_namespace(config), config


class Prompter:

    def __init__(self, prompt_type="appear_details", min_tokens=256, max_tokens=1280):

        self.prompt = {
            "binary": binary_prompt,
            "ternary": ternary_prompt,
            "decimal": decimal_prompt,
            "object": object_prompt,
            "object_yesno": object_yesno_prompt,
            "object_details": object_details_prompt,
            "object_simple": object_simple_prompt,
            "class": class_prompt,
            "class_simple": class_simple_prompt,
            "landmark": landmark_prompt,
            "met": met_prompt,
        }[prompt_type]

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            min_pixels=min_tokens * 28 * 28,
            max_pixels=max_tokens * 28 * 28,
            use_fast=False,
        )
        self.text_template = self.get_text_template()

    def get_text_template(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    # {"type": "image", "image": query_file_path, "resized_height": query_resized_height, "resized_width": query_resized_width},
                    # {"type": "image", "image": shortlisted_file_path, "resized_height": shortlisted_resized_height, "resized_width": shortlisted_resized_width},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text

    def image_processor(self, image):
        return self.processor.image_processor(images=[image], return_tensors="pt")

    def batch_encode(self, query_image, db_image):
        image_grid_thw = torch.cat([query_image["image_grid_thw"], db_image["image_grid_thw"]])
        image_embeds = torch.cat([query_image["image_embeds"], db_image["image_embeds"]])
        text = [self.text_template]
        merge_length = self.processor.image_processor.merge_size ** 2
        index = 0
        for i in range(len(text)):
            while self.processor.image_token in text[i]:
                num_image_tokens = image_grid_thw[index].prod() // merge_length
                text[i] = text[i].replace(self.processor.image_token, "<|placeholder|>" * num_image_tokens, 1)
                index += 1
            text[i] = text[i].replace("<|placeholder|>", self.processor.image_token)
        text_inputs = self.processor.tokenizer(text, padding=True)
        #self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])
        return BatchFeature(data={**text_inputs, "image_embeds": image_embeds, "image_grid_thw": image_grid_thw}, tensor_type="pt")

    def batch_decode(self, *args, **kwargs):
        return self.processor.batch_decode(*args, **kwargs)
