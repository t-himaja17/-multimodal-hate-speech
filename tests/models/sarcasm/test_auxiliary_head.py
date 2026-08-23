import pytest
import torch

from multimodal_hate.models.sarcasm.auxiliary_head import (
    SarcasmAuxiliaryHead,
)


def test_output_shape():
    head = SarcasmAuxiliaryHead(input_dim=16)

    output = head(torch.randn(4, 16))

    assert output.shape == (4, 1)


def test_single_sample():
    head = SarcasmAuxiliaryHead(input_dim=16)

    output = head(torch.randn(1, 16))

    assert output.shape == (1, 1)


def test_batch_support():
    head = SarcasmAuxiliaryHead(input_dim=32)

    output = head(torch.randn(10, 32))

    assert output.shape == (10, 1)


def test_hidden_layer():
    head = SarcasmAuxiliaryHead(
        input_dim=32,
        hidden_dim=16,
    )

    output = head(torch.randn(5, 32))

    assert output.shape == (5, 1)


def test_probability_range():
    head = SarcasmAuxiliaryHead(input_dim=8)

    logits = head(torch.randn(5, 8))
    probabilities = head.probability(logits)

    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_bce_compatibility():
    head = SarcasmAuxiliaryHead(input_dim=8)

    logits = head(torch.randn(5, 8))
    targets = torch.randint(
        0,
        2,
        (5, 1),
    ).float()

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gradients():
    head = SarcasmAuxiliaryHead(input_dim=8)

    x = torch.randn(
        4,
        8,
        requires_grad=True,
    )

    logits = head(x)
    loss = logits.sum()
    loss.backward()

    assert x.grad is not None

    for parameter in head.parameters():
        assert parameter.grad is not None


def test_invalid_input_dimension():
    with pytest.raises(ValueError):
        SarcasmAuxiliaryHead(input_dim=0)


def test_wrong_input_shape():
    head = SarcasmAuxiliaryHead(input_dim=8)

    with pytest.raises(ValueError):
        head(torch.randn(4, 8, 2))


def test_wrong_feature_dimension():
    head = SarcasmAuxiliaryHead(input_dim=8)

    with pytest.raises(ValueError):
        head(torch.randn(4, 16))


def test_non_tensor_input():
    head = SarcasmAuxiliaryHead(input_dim=8)

    with pytest.raises(TypeError):
        head([[1.0] * 8])


def test_empty_batch():
    head = SarcasmAuxiliaryHead(input_dim=8)

    with pytest.raises(ValueError):
        head(torch.empty(0, 8))


def test_probability_invalid_shape():
    with pytest.raises(ValueError):
        SarcasmAuxiliaryHead.probability(
            torch.randn(4, 2)
        )