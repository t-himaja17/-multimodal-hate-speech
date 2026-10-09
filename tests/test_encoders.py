import torch

from src.multimodal_hate.models.encoders.base import BaseEncoder


class DummyEncoder(BaseEncoder):
    """Small encoder used to test the common encoder interface."""

    def __init__(self, output_dim=64):
        super().__init__(output_dim=output_dim)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        return torch.zeros(batch_size, self.output_dim)


def test_base_encoder_stores_output_dimension():
    encoder = DummyEncoder(output_dim=64)

    assert encoder.output_dim == 64


def test_base_encoder_returns_tensor():
    encoder = DummyEncoder(output_dim=64)

    inputs = torch.randn(2, 10)
    output = encoder(inputs)

    assert isinstance(output, torch.Tensor)


def test_base_encoder_preserves_batch_size():
    encoder = DummyEncoder(output_dim=64)

    inputs = torch.randn(4, 10)
    output = encoder(inputs)

    assert output.shape[0] == 4


def test_base_encoder_output_dimension():
    encoder = DummyEncoder(output_dim=64)

    inputs = torch.randn(3, 10)
    output = encoder(inputs)

    assert output.shape == (3, 64)


# ------------------------------------------------------------------
# ViT-L/14 Encoder
# ------------------------------------------------------------------

def test_vit_encoder_output_shape():
    from src.multimodal_hate.models.encoders.vit import ViTEncoder

    encoder = ViTEncoder()

    images = torch.randn(1, 3, 224, 224)
    output = encoder(images)

    assert output.shape == (1, 257, 1024)


def test_vit_encoder_output_dimension():
    from src.multimodal_hate.models.encoders.vit import ViTEncoder

    encoder = ViTEncoder()

    assert encoder.output_dim == 1024


# ------------------------------------------------------------------
# CLIP Image Encoder
# ------------------------------------------------------------------

def test_clip_image_encoder_output_shape():
    from src.multimodal_hate.models.encoders.clip import CLIPImageEncoder

    encoder = CLIPImageEncoder()

    images = torch.randn(1, 3, 224, 224)
    output = encoder(images)

    assert output.shape == (1, 512)


def test_clip_image_encoder_output_dimension():
    from src.multimodal_hate.models.encoders.clip import CLIPImageEncoder

    encoder = CLIPImageEncoder()

    assert encoder.output_dim == 512


# ------------------------------------------------------------------
# CLIP Text Encoder
# ------------------------------------------------------------------

def test_clip_text_encoder_output_shape():
    from src.multimodal_hate.models.encoders.clip import CLIPTextEncoder

    encoder = CLIPTextEncoder()

    output = encoder(["This is a test"])

    assert output.shape == (1, 512)


def test_clip_text_encoder_output_dimension():
    from src.multimodal_hate.models.encoders.clip import CLIPTextEncoder

    encoder = CLIPTextEncoder()

    assert encoder.output_dim == 512


# ------------------------------------------------------------------
# HateBERT Encoder
# ------------------------------------------------------------------

def test_hatebert_encoder_output_shape():
    from src.multimodal_hate.models.encoders.hatebert import HateBERTEncoder

    encoder = HateBERTEncoder()

    output = encoder(["This is a test"])

    assert output.shape == (1, 768)


def test_hatebert_encoder_output_dimension():
    from src.multimodal_hate.models.encoders.hatebert import HateBERTEncoder

    encoder = HateBERTEncoder()

    assert encoder.output_dim == 768


# ------------------------------------------------------------------
# RoBERTa-large Encoder
# ------------------------------------------------------------------

def test_roberta_encoder_output_shape():
    from src.multimodal_hate.models.encoders.roberta import RoBERTaEncoder

    encoder = RoBERTaEncoder()

    output = encoder(["This is a test"])

    assert output.shape == (1, 1024)


def test_roberta_encoder_output_dimension():
    from src.multimodal_hate.models.encoders.roberta import RoBERTaEncoder

    encoder = RoBERTaEncoder()

    assert encoder.output_dim == 1024