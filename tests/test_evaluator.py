import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DummyModel(nn.Module):
    """Small model matching the evaluator's expected output."""

    def __init__(self):
        super().__init__()

        self.bias = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(self, images, texts):
        batch_size = images.size(0)

        hate_logits = torch.tensor(
            [
                [-2.0, 2.0],
                [2.0, -2.0],
                [-2.0, 2.0],
                [2.0, -2.0],
            ],
            device=images.device,
        )[:batch_size]

        sarcasm_logits = torch.tensor(
            [
                [2.0, -2.0],
                [-2.0, 2.0],
                [2.0, -2.0],
                [-2.0, 2.0],
            ],
            device=images.device,
        )[:batch_size]

        target_logits = torch.zeros(
            batch_size,
            5,
            device=images.device,
        )

        return {
            "logits": {
                "hate": hate_logits,
                "sarcasm": sarcasm_logits,
                "target": target_logits,
            }
        }


def make_batch():
    return {
        "image": torch.randn(
            4,
            3,
            224,
            224,
        ),
        "text": [
            "sample one",
            "sample two",
            "sample three",
            "sample four",
        ],
        "hate_target": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        "sarcasm_target": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        "target_target": torch.zeros(
            4,
            5,
        ),
        "sample_id": [
            "1",
            "2",
            "3",
            "4",
        ],
    }


class SingleBatchLoader:
    """Minimal DataLoader-like iterable for evaluator tests."""

    def __iter__(self):
        yield make_batch()


def test_evaluator_runs():
    from multimodal_hate.evaluation.evaluator import (
        MultimodalEvaluator,
    )

    model = DummyModel()

    evaluator = MultimodalEvaluator(
        model,
        device="cpu",
    )

    results = evaluator.evaluate(
        SingleBatchLoader()
    )

    assert "metrics" in results
    assert "predictions" in results
    assert "targets" in results


def test_evaluator_returns_binary_predictions():
    from multimodal_hate.evaluation.evaluator import (
        MultimodalEvaluator,
    )

    model = DummyModel()

    evaluator = MultimodalEvaluator(
        model,
        device="cpu",
    )

    results = evaluator.evaluate(
        SingleBatchLoader()
    )

    hate_probabilities = results[
        "predictions"
    ]["hate"]

    sarcasm_probabilities = results[
        "predictions"
    ]["sarcasm"]

    assert hate_probabilities.shape == (4,)
    assert sarcasm_probabilities.shape == (4,)

    assert torch.all(
        hate_probabilities >= 0
    )

    assert torch.all(
        hate_probabilities <= 1
    )

    assert torch.all(
        sarcasm_probabilities >= 0
    )

    assert torch.all(
        sarcasm_probabilities <= 1
    )


def test_evaluator_returns_target_probabilities():
    from multimodal_hate.evaluation.evaluator import (
        MultimodalEvaluator,
    )

    model = DummyModel()

    evaluator = MultimodalEvaluator(
        model,
        device="cpu",
    )

    results = evaluator.evaluate(
        SingleBatchLoader()
    )

    target_probabilities = results[
        "predictions"
    ]["target"]

    assert target_probabilities.shape == (
        4,
        5,
    )

    assert torch.all(
        target_probabilities >= 0
    )

    assert torch.all(
        target_probabilities <= 1
    )


def test_evaluator_returns_targets():
    from multimodal_hate.evaluation.evaluator import (
        MultimodalEvaluator,
    )

    evaluator = MultimodalEvaluator(
        DummyModel(),
        device="cpu",
    )

    results = evaluator.evaluate(
        SingleBatchLoader()
    )

    assert results["targets"]["hate"].shape == (4,)
    assert results["targets"]["sarcasm"].shape == (4,)
    assert results["targets"]["target"].shape == (4, 5)


def test_evaluator_metrics_are_available():
    from multimodal_hate.evaluation.evaluator import (
        MultimodalEvaluator,
    )

    evaluator = MultimodalEvaluator(
        DummyModel(),
        device="cpu",
    )

    results = evaluator.evaluate(
        SingleBatchLoader()
    )

    hate_metrics = results["metrics"]["hate"]
    sarcasm_metrics = results["metrics"]["sarcasm"]

    expected = {
        "AUROC",
        "Macro_F1",
        "Balanced_Accuracy",
        "False_Positive_Rate",
        "Cohen_Kappa",
    }

    assert set(hate_metrics.keys()) == expected
    assert set(sarcasm_metrics.keys()) == expected