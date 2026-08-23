import torch
from torch import nn

from multimodal_hate.models.multimodal_model import (
    MultimodalHateSpeechModel,
)


class DummyImageEncoder(nn.Module):
    output_dim = 1024

    def forward(self, images):
        batch_size = images.size(0)

        return torch.randn(
            batch_size,
            4,
            1024,
        )


class DummyTextEncoder(nn.Module):
    output_dim = 768

    def encode_tokens(self, texts):
        batch_size = len(texts)

        return torch.randn(
            batch_size,
            6,
            768,
        )


class DummyCLIPImageEncoder(nn.Module):
    output_dim = 512

    def forward(self, images):
        return torch.randn(
            images.size(0),
            512,
        )


class DummyCLIPTextEncoder(nn.Module):
    output_dim = 512

    def forward(self, texts):
        return torch.randn(
            len(texts),
            512,
        )


class DummySarcasmEncoder(nn.Module):
    def forward(self, texts):
        return torch.full(
            (len(texts), 1),
            0.5,
        )


class DummySentimentModule(nn.Module):
    def forward(self, texts):
        return torch.zeros(
            len(texts),
            1,
        )


def create_model():
    return MultimodalHateSpeechModel(
        image_encoder=DummyImageEncoder(),
        text_encoder=DummyTextEncoder(),
        clip_image_encoder=DummyCLIPImageEncoder(),
        clip_text_encoder=DummyCLIPTextEncoder(),
        sarcasm_encoder=DummySarcasmEncoder(),
        sentiment_module=DummySentimentModule(),
        fusion_dim=512,
        sarcasm_gate_dim=64,
        num_fusion_layers=2,
        num_heads=8,
        dropout=0.1,
    )


def test_full_model_forward():
    model = create_model()

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    texts = [
        "test meme one",
        "test meme two",
    ]

    output = model(
        images,
        texts,
    )

    assert output.fused_tokens.shape == (
        2,
        10,
        512,
    )

    assert output.pooled.shape == (
        2,
        512,
    )

    assert output.image_representation.shape == (
        2,
        512,
    )

    assert output.text_representation.shape == (
        2,
        512,
    )

    assert output.sarcasm_gate.shape == (
        2,
        64,
    )

    assert output.sarcasm_probability.shape == (
        2,
        1,
    )

    assert output.incongruity_score.shape == (
        2,
        1,
    )

    assert output.sentiment_reversal.shape == (
        2,
        1,
    )

    assert output.logits["hate"].shape == (
        2,
        2,
    )

    assert output.logits["sarcasm"].shape == (
        2,
        2,
    )

    assert output.logits["target"].shape == (
        2,
        5,
    )


def test_model_accepts_external_sarcasm_signals():
    model = create_model()

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    texts = [
        "first",
        "second",
    ]

    sarcasm_probability = torch.tensor(
        [
            [0.9],
            [0.1],
        ]
    )

    sentiment_reversal = torch.tensor(
        [
            [1.0],
            [0.0],
        ]
    )

    output = model(
        images,
        texts,
        sarcasm_probability=sarcasm_probability,
        sentiment_reversal=sentiment_reversal,
    )

    assert torch.equal(
        output.sarcasm_probability,
        sarcasm_probability,
    )

    assert torch.equal(
        output.sentiment_reversal,
        sentiment_reversal,
    )