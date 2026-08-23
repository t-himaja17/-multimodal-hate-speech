from PIL import Image

from src.multimodal_hate.data.processing.image import validate_image


def test_validate_valid_image(tmp_path):
    image_path = tmp_path / "test.png"

    image = Image.new("RGB", (100, 100))
    image.save(image_path)

    assert validate_image(str(image_path)) is True


def test_validate_missing_image():
    assert validate_image("does_not_exist.png") is False


def test_validate_invalid_image(tmp_path):
    image_path = tmp_path / "invalid.png"
    image_path.write_bytes(b"not a real image")

    assert validate_image(str(image_path)) is False


def test_validate_none():
    assert validate_image(None) is False