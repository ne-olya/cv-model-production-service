from io import BytesIO

import pytest
from PIL import Image

from app.preprocessing import InvalidImage, decode_image, preprocess


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (32, 20), "red").save(buffer, "PNG")
    return buffer.getvalue()


def test_preprocess_shape():
    assert preprocess(decode_image(image_bytes(), 100_000), 64).shape == (3, 64, 64)


def test_invalid_image():
    with pytest.raises(InvalidImage):
        decode_image(b"not an image", 100_000)
