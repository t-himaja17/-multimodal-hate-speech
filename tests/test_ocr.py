
import pytest

from src.multimodal_hate.data.processing.ocr import (
    OCRResult,
    create_ocr_result,
    format_ocr_text,
)


def test_create_ocr_result():
    result = create_ocr_result(
        text="Example meme text",
        bounding_box=(10, 20, 200, 100),
        region="TOP",
        confidence=0.95,
    )

    assert isinstance(result, OCRResult)
    assert result.text == "Example meme text"
    assert result.bounding_box == (10, 20, 200, 100)
    assert result.region == "TOP"
    assert result.confidence == 0.95


def test_create_ocr_result_without_optional_fields():
    result = create_ocr_result(
        text="Example text",
        bounding_box=(0, 0, 100, 100),
    )

    assert result.text == "Example text"
    assert result.region is None
    assert result.confidence is None


def test_create_ocr_result_rejects_empty_text():
    with pytest.raises(ValueError):
        create_ocr_result(
            text="",
            bounding_box=(0, 0, 100, 100),
        )


def test_create_ocr_result_rejects_invalid_bounding_box():
    with pytest.raises(ValueError):
        create_ocr_result(
            text="Example text",
            bounding_box=(0, 0, 100),
        )


def test_create_ocr_result_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        create_ocr_result(
            text="Example text",
            bounding_box=(0, 0, 100, 100),
            confidence=1.5,
        )


def test_format_ocr_text_with_regions():
    results = [
        create_ocr_result(
            text="Top text",
            bounding_box=(0, 0, 100, 50),
            region="TOP",
            confidence=0.95,
        ),
        create_ocr_result(
            text="Bottom text",
            bounding_box=(0, 50, 100, 100),
            region="BOTTOM",
            confidence=0.90,
        ),
    ]

    formatted = format_ocr_text(results)

    assert formatted == "[TOP] Top text [BOTTOM] Bottom text"


def test_format_ocr_text_with_caption():
    results = [
        create_ocr_result(
            text="Caption text",
            bounding_box=(0, 0, 100, 50),
            region="CAPTION",
            confidence=0.88,
        )
    ]

    formatted = format_ocr_text(results)

    assert formatted == "[CAPTION] Caption text"


def test_format_ocr_text_with_empty_results():
    assert format_ocr_text([]) == ""