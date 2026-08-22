import torch
from torch import nn

from multimodal_hate.models.sarcasm.sarcasm_bert import SarcasmBERT


class DummyTokenizer:
    """Small deterministic tokenizer for unit tests."""

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ):
        batch_size = len(texts)

        return {
            "input_ids": torch.ones(
                batch_size,
                4,
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                batch_size,
                4,
                dtype=torch.long,
            ),
        }


class DummyConfig:
    hidden_size = 8


class DummyTransformer(nn.Module):
    """Small Transformer-like model for tests."""

    def __init__(self):
        super().__init__()

        self.config = DummyConfig()

        self.projection = nn.Linear(
            4,
            8,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        # Convert token IDs to float and project them.
        x = input_ids.float()

        x = self.projection(
            x.unsqueeze(-1).repeat(1, 1, 4)
        )

        return type(
            "DummyOutput",
            (),
            {
                "last_hidden_state": x
            },
        )()


def create_model():
    """Create SarcasmBERT without downloading a real model."""

    transformer = DummyTransformer()
    tokenizer = DummyTokenizer()

    return SarcasmBERT(
        model_name="test-model",
        device="cpu",
        max_length=16,
        dropout=0.0,
        pretrained_model=transformer,
        tokenizer=tokenizer,
    )


def test_single_text_shape_and_range():
    model = create_model()

    result = model(
        "This is a test sentence."
    )

    assert result.probability.shape == (1, 1)
    assert result.representation.shape == (1, 8)

    assert torch.all(
        result.probability >= 0.0
    )

    assert torch.all(
        result.probability <= 1.0
    )


def test_batch_behavior():
    model = create_model()

    texts = [
        "This is the first text.",
        "This is the second text.",
        "This is the third text.",
    ]

    result = model(texts)

    assert result.probability.shape == (3, 1)
    assert result.representation.shape == (3, 8)


def test_probability_matches_sigmoid_of_logits():
    model = create_model()

    result = model(
        [
            "Test one.",
            "Test two.",
        ]
    )

    # Reconstruct logits from the returned representation.
    logits = model.classifier(
        result.representation
    )

    expected_probability = torch.sigmoid(
        logits
    )

    assert torch.allclose(
        result.probability,
        expected_probability,
    )


def test_gradients_are_preserved():
    model = create_model()

    result = model(
        [
            "Gradient test one.",
            "Gradient test two.",
        ]
    )

    loss = result.probability.sum()

    loss.backward()

    assert model.classifier.weight.grad is not None

    assert model.transformer.projection.weight.grad is not None


def test_probability_is_deterministic_without_dropout():
    torch.manual_seed(42)

    model = create_model()

    texts = [
        "Deterministic test.",
        "Another deterministic test.",
    ]

    result_one = model(texts)

    result_two = model(texts)

    assert torch.allclose(
        result_one.probability,
        result_two.probability,
    )


def test_span_markers_are_not_fabricated():
    model = create_model()

    result = model(
        "Span test."
    )

    assert result.span_markers is None


def test_empty_batch_is_rejected():
    model = create_model()

    try:
        model([])

        assert False, (
            "Expected ValueError for empty input."
        )

    except ValueError:
        pass


def test_non_string_input_is_rejected():
    model = create_model()

    try:
        model([123])

        assert False, (
            "Expected TypeError for non-string input."
        )

    except TypeError:
        pass