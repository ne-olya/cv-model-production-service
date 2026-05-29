from io import BytesIO

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


class InvalidImage(ValueError):
    pass


def decode_image(content: bytes, max_bytes: int):
    if not content:
        raise InvalidImage("Empty file")
    if len(content) > max_bytes:
        raise InvalidImage("File is too large")
    try:
        image = Image.open(BytesIO(content))
        image.verify()
        image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImage("Unsupported or corrupted image") from exc
    return image


def preprocess(image: Image.Image, size=224):
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std
