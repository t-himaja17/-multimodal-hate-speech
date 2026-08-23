import pytest
import torch

from multimodal_hate.models.fusion.projection import (
    ModalityProjection,
    ProjectionAlignment,
)


def test_image_projection_shape():
    """1024-dimensional image tokens should become 512-dimensional."""
    module = ModalityProjection(
        input_dim=1024,
        fusion_dim=512,
    )

    x = torch.randn(2, 257, 1024)

    output = module(x)

    assert output.shape == (2, 257, 512)


def test_text_projection_shape():
    """768-dimensional text tokens should become 512-dimensional."""
    module = ModalityProjection(
        input_dim=768,
        fusion_dim=512,
    )

    x = torch.randn(2, 128, 768)

    output = module(x)

    assert output.shape == (2, 128, 512)


def test_projection_alignment_shapes():
    """Image and text should both enter the shared 512-D fusion space."""
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(4, 257, 1024)
    text = torch.randn(4, 128, 768)

    projected_image, projected_text = module(
        image,
        text,
    )

    assert projected_image.shape == (4, 257, 512)
    assert projected_text.shape == (4, 128, 512)


def test_projection_alignment_supports_different_sequence_lengths():
    """N_img and N_txt must remain independently variable."""
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(2, 10, 1024)
    text = torch.randn(2, 37, 768)

    projected_image, projected_text = module(
        image,
        text,
    )

    assert projected_image.shape == (2, 10, 512)
    assert projected_text.shape == (2, 37, 512)


def test_modality_embeddings_are_learnable():
    """IMG_TOKEN and TXT_TOKEN must be trainable parameters."""
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    assert isinstance(module.img_token, torch.nn.Parameter)
    assert isinstance(module.txt_token, torch.nn.Parameter)

    assert module.img_token.requires_grad
    assert module.txt_token.requires_grad


def test_projection_gradients_flow():
    """Gradients must flow through both projection branches."""
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(
        2,
        10,
        1024,
        requires_grad=True,
    )

    text = torch.randn(
        2,
        12,
        768,
        requires_grad=True,
    )

    projected_image, projected_text = module(
        image,
        text,
    )

    loss = projected_image.mean() + projected_text.mean()

    loss.backward()

    assert image.grad is not None
    assert text.grad is not None

    assert module.img_token.grad is not None
    assert module.txt_token.grad is not None


def test_projection_rejects_wrong_image_dimension():
    """Incorrect image feature dimensions must fail clearly."""
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(2, 10, 768)
    text = torch.randn(2, 10, 768)

    with pytest.raises(ValueError):
        module(image, text)


def test_projection_rejects_wrong_text_dimension():
    """Incorrect text feature dimensions must fail clearly."""
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(2, 10, 1024)
    text = torch.randn(2, 10, 512)

    with pytest.raises(ValueError):
        module(image, text)


def test_projection_rejects_batch_mismatch():
    """Image and text batch sizes must match."""
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(2, 10, 1024)
    text = torch.randn(3, 10, 768)

    with pytest.raises(ValueError):
        module(image, text)


def test_projection_rejects_non_3d_image_input():
    """Image tokens must have [B, N, D] shape."""
    module = ModalityProjection(
        input_dim=1024,
        fusion_dim=512,
    )

    image = torch.randn(2, 1024)

    with pytest.raises(ValueError):
        module(image)