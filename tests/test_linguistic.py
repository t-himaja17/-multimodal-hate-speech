import torch

from src.multimodal_hate.models.features.linguistic import (
    extract_linguistic_features,
)


def test_linguistic_features_output_shape():
    output = extract_linguistic_features(
        ["This is a very good example."]
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 64)


def test_linguistic_features_batch_shape():
    output = extract_linguistic_features(
        [
            "This is good.",
            "This is not good.",
            "This is really bad!",
        ]
    )

    assert output.shape == (3, 64)


def test_linguistic_features_dtype():
    output = extract_linguistic_features(
        ["This is a test."]
    )

    assert output.dtype == torch.float32


def test_negation_features():
    output = extract_linguistic_features(
        ["This is not good."]
    )

    assert output[0, 8] >= 1
    assert output[0, 10] == 1


def test_intensifier_features():
    output = extract_linguistic_features(
        ["This is very good."]
    )

    assert output[0, 16] >= 1
    assert output[0, 18] == 1


def test_sentiment_features():
    output = extract_linguistic_features(
        ["This is good but terrible."]
    )

    assert output[0, 24] >= 1
    assert output[0, 25] >= 1


def test_punctuation_features():
    output = extract_linguistic_features(
        ["What!!!"]
    )

    assert output[0, 32] == 3


def test_single_string_input():
    output = extract_linguistic_features("This is a test.")

    assert output.shape == (1, 64)
