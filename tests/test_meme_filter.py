import pytest

from src.multimodal_hate.data.processing.meme_filter import (
    MemeClassificationResult,
    create_meme_classification_result,
    is_meme_image,
)


def test_create_meme_classification_result():
    result = create_meme_classification_result(
        is_meme=True,
        confidence=0.95,
    )

    assert isinstance(result, MemeClassificationResult)
    assert result.is_meme is True
    assert result.confidence == 0.95
    assert result.model_name == "ResNet-50"
    assert result.source == "Know Your Meme"


def test_create_non_meme_result():
    result = create_meme_classification_result(
        is_meme=False,
        confidence=0.90,
    )

    assert result.is_meme is False
    assert result.confidence == 0.90


def test_invalid_confidence():
    with pytest.raises(ValueError):
        create_meme_classification_result(
            is_meme=True,
            confidence=1.5,
        )


def test_empty_model_name():
    with pytest.raises(ValueError):
        create_meme_classification_result(
            is_meme=True,
            confidence=0.9,
            model_name="",
        )


def test_empty_source():
    with pytest.raises(ValueError):
        create_meme_classification_result(
            is_meme=True,
            confidence=0.9,
            source="",
        )


def test_meme_image_above_threshold():
    result = create_meme_classification_result(
        is_meme=True,
        confidence=0.95,
    )

    assert is_meme_image(result, threshold=0.5) is True


def test_meme_image_below_threshold():
    result = create_meme_classification_result(
        is_meme=True,
        confidence=0.40,
    )

    assert is_meme_image(result, threshold=0.5) is False


def test_non_meme_is_rejected():
    result = create_meme_classification_result(
        is_meme=False,
        confidence=0.95,
    )

    assert is_meme_image(result, threshold=0.5) is False


def test_invalid_threshold():
    result = create_meme_classification_result(
        is_meme=True,
        confidence=0.95,
    )

    with pytest.raises(ValueError):
        is_meme_image(result, threshold=1.5)