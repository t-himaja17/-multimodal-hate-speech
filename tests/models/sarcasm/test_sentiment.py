import torch
import pytest

from multimodal_hate.models.sarcasm.sentiment import (
    SentimentReversal,
)


class DummyTokenizer:
    """Small tokenizer used so tests never download a BERT tokenizer."""

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ):
        if isinstance(texts, str):
            texts = [texts]

        batch_size = len(texts)

        return {
            "input_ids": torch.ones(
                batch_size,
                8,
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                batch_size,
                8,
                dtype=torch.long,
            ),
        }


class DummyOutput:
    def __init__(self, logits):
        self.logits = logits


class DummySentimentModel(torch.nn.Module):
    """Deterministic fake BERT sentiment model."""

    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Linear(8, 2)

        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(
                torch.tensor([0.0, 1.0])
            )

    def forward(self, input_ids, attention_mask):
        pooled = input_ids.float().mean(dim=1)

        features = pooled.unsqueeze(1).repeat(1, 8)

        return DummyOutput(
            self.linear(features)
        )


@pytest.fixture
def sentiment_model():
    return SentimentReversal(
        tokenizer=DummyTokenizer(),
        model=DummySentimentModel(),
        device="cpu",
    )


def test_single_text_output_shape(sentiment_model):
    output = sentiment_model(
        "I absolutely love this."
    )

    assert output.shape == (1, 1)


def test_batch_output_shape(sentiment_model):
    texts = [
        "I love this.",
        "This is terrible.",
        "This is okay.",
    ]

    output = sentiment_model(texts)

    assert output.shape == (3, 1)


def test_reversal_output_range(sentiment_model):
    output = sentiment_model(
        [
            "I love this.",
            "I hate this.",
        ]
    )

    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_vader_scores_shape(sentiment_model):
    scores = sentiment_model.vader_scores(
        [
            "I love this.",
            "I hate this.",
        ]
    )

    assert scores.shape == (2, 1)


def test_vader_positive_negative(sentiment_model):
    positive = sentiment_model.vader_scores(
        "I absolutely love this wonderful product."
    )

    negative = sentiment_model.vader_scores(
        "I absolutely hate this terrible product."
    )

    assert positive.item() > 0.0
    assert negative.item() < 0.0


def test_region_aware_processing(sentiment_model):
    output = sentiment_model(
        region_texts={
            "TOP": [
                "I love this.",
                "This is amazing.",
            ],
            "BOTTOM": [
                "This is terrible.",
                "I hate this.",
            ],
        }
    )

    assert output.shape == (2, 1)


def test_top_region_only(sentiment_model):
    output = sentiment_model(
        region_texts={
            "TOP": [
                "I love this.",
                "I hate this.",
            ]
        }
    )

    assert output.shape == (2, 1)


def test_bottom_region_only(sentiment_model):
    output = sentiment_model(
        region_texts={
            "BOTTOM": [
                "I love this.",
                "I hate this.",
            ]
        }
    )

    assert output.shape == (2, 1)


def test_region_batch_mismatch_rejected(sentiment_model):
    with pytest.raises(ValueError):
        sentiment_model(
            region_texts={
                "TOP": [
                    "one",
                    "two",
                ],
                "BOTTOM": [
                    "only one",
                ],
            }
        )


def test_empty_input_rejected(sentiment_model):
    with pytest.raises(ValueError):
        sentiment_model([])


def test_invalid_text_type_rejected(sentiment_model):
    with pytest.raises(TypeError):
        sentiment_model([123, 456])


def test_both_text_inputs_rejected(sentiment_model):
    with pytest.raises(ValueError):
        sentiment_model(
            texts=["hello"],
            region_texts={
                "TOP": ["hello"]
            },
        )


def test_no_text_input_rejected(sentiment_model):
    with pytest.raises(ValueError):
        sentiment_model()


def test_gradients_exist_in_bert_model(sentiment_model):
    loss = sentiment_model.model.linear.weight.sum()

    loss.backward()

    assert sentiment_model.model.linear.weight.grad is not None


def test_deterministic_vader(sentiment_model):
    text = "This is surprisingly wonderful!"

    first = sentiment_model.vader_scores(text)
    second = sentiment_model.vader_scores(text)

    assert torch.equal(first, second)


def test_binary_reversal_values(sentiment_model):
    vader = torch.tensor(
        [[1.0], [-1.0], [1.0], [-1.0]]
    )

    bert = torch.tensor(
        [[1.0], [1.0], [-1.0], [-1.0]]
    )

    reversal = sentiment_model._calculate_reversal(
        vader,
        bert,
    )

    expected = torch.tensor(
        [[0.0], [1.0], [1.0], [0.0]]
    )

    assert torch.equal(
        reversal,
        expected,
    )


def test_polarity_shape(sentiment_model):
    polarity = sentiment_model.bert_polarity(
        [
            "positive",
            "negative",
            "neutral",
        ]
    )

    assert polarity.shape == (3, 1)


def test_invalid_model_output_rejected():
    class BadModel(torch.nn.Module):
        def forward(self, **kwargs):
            return torch.ones(2, 3, 4)

    with pytest.raises(ValueError):
        sentiment = SentimentReversal(
            tokenizer=DummyTokenizer(),
            model=BadModel(),
        )

        sentiment.bert_probabilities(
            ["hello", "world"]
        )