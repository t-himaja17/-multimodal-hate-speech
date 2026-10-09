import pytest
import torch

from multimodal_hate.models.sarcasm.incongruity import (
    VisualTextIncongruity,
)


def create_model():
    return VisualTextIncongruity(
        embedding_dim=512,
        hidden_dim=16,
    )


def test_output_shape_and_range():
    model = create_model()

    image = torch.randn(4, 512)
    text = torch.randn(4, 512)

    output = model(image, text)

    assert output.shape == (4, 1)
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_single_batch():
    model = create_model()

    image = torch.randn(1, 512)
    text = torch.randn(1, 512)

    output = model(image, text)

    assert output.shape == (1, 1)


def test_cosine_similarity_identical_vectors():
    model = create_model()

    image = torch.randn(3, 512)
    text = image.clone()

    similarity = model.cosine_similarity(
        image,
        text,
    )

    assert similarity.shape == (3, 1)
    assert torch.allclose(
        similarity,
        torch.ones_like(similarity),
        atol=1e-5,
    )


def test_dissimilarity_identical_vectors_is_zero():
    model = create_model()

    image = torch.randn(3, 512)
    text = image.clone()

    dissimilarity = model.semantic_dissimilarity(
        image,
        text,
    )

    assert dissimilarity.shape == (3, 1)
    assert torch.allclose(
        dissimilarity,
        torch.zeros_like(dissimilarity),
        atol=1e-5,
    )


def test_opposite_vectors_have_high_dissimilarity():
    model = create_model()

    image = torch.randn(3, 512)
    text = -image

    dissimilarity = model.semantic_dissimilarity(
        image,
        text,
    )

    assert torch.allclose(
        dissimilarity,
        torch.full_like(dissimilarity, 2.0),
        atol=1e-5,
    )


def test_zero_vectors_do_not_produce_nan():
    model = create_model()

    image = torch.zeros(2, 512)
    text = torch.zeros(2, 512)

    similarity = model.cosine_similarity(
        image,
        text,
    )

    output = model(image, text)

    assert torch.isfinite(similarity).all()
    assert torch.isfinite(output).all()


def test_gradients_are_preserved():
    model = create_model()

    image = torch.randn(
        2,
        512,
        requires_grad=True,
    )

    text = torch.randn(
        2,
        512,
        requires_grad=True,
    )

    output = model(image, text)

    loss = output.sum()
    loss.backward()

    assert image.grad is not None
    assert text.grad is not None

    for parameter in model.parameters():
        assert parameter.grad is not None


def test_deterministic_mathematical_operation():
    model = create_model()

    image = torch.randn(2, 512)
    text = torch.randn(2, 512)

    result_one = model.semantic_dissimilarity(
        image,
        text,
    )

    result_two = model.semantic_dissimilarity(
        image,
        text,
    )

    assert torch.allclose(
        result_one,
        result_two,
    )


def test_dimension_mismatch_is_rejected():
    model = create_model()

    image = torch.randn(2, 512)
    text = torch.randn(2, 768)

    with pytest.raises(ValueError):
        model(image, text)


def test_batch_mismatch_is_rejected():
    model = create_model()

    image = torch.randn(2, 512)
    text = torch.randn(3, 512)

    with pytest.raises(ValueError):
        model(image, text)


def test_device_mismatch_is_rejected():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    model = create_model()

    image = torch.randn(2, 512, device="cpu")
    text = torch.randn(2, 512, device="cuda")

    with pytest.raises(ValueError):
        model(image, text)


def test_wrong_rank_is_rejected():
    model = create_model()

    image = torch.randn(512)
    text = torch.randn(512)

    with pytest.raises(ValueError):
        model(image, text)


def test_non_tensor_input_is_rejected():
    model = create_model()

    image = [[1.0] * 512]
    text = torch.randn(1, 512)

    with pytest.raises(TypeError):
        model(image, text)