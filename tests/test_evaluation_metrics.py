import math

import numpy as np
import torch

from multimodal_hate.evaluation.metrics import (
    auroc,
    balanced_accuracy,
    binary_predictions,
    cohen_kappa,
    evaluate_binary_classification,
    false_positive_rate,
    macro_f1,
)


def test_binary_predictions():
    probabilities = torch.tensor(
        [0.1, 0.6, 0.8, 0.2]
    )

    predictions = binary_predictions(
        probabilities
    )

    assert np.array_equal(
        predictions,
        np.array([0, 1, 1, 0]),
    )


def test_auroc():
    targets = torch.tensor(
        [0, 0, 1, 1]
    )

    probabilities = torch.tensor(
        [0.1, 0.2, 0.8, 0.9]
    )

    score = auroc(
        targets,
        probabilities,
    )

    assert score == 1.0


def test_macro_f1():
    targets = np.array(
        [0, 0, 1, 1]
    )

    predictions = np.array(
        [0, 1, 1, 1]
    )

    score = macro_f1(
        targets,
        predictions,
    )

    assert 0.0 <= score <= 1.0


def test_balanced_accuracy():
    targets = np.array(
        [0, 0, 1, 1]
    )

    predictions = np.array(
        [0, 0, 1, 1]
    )

    score = balanced_accuracy(
        targets,
        predictions,
    )

    assert score == 1.0


def test_false_positive_rate():
    targets = np.array(
        [0, 0, 0, 1, 1]
    )

    predictions = np.array(
        [0, 1, 0, 1, 0]
    )

    score = false_positive_rate(
        targets,
        predictions,
    )

    assert score == 1.0 / 3.0


def test_cohen_kappa():
    targets = np.array(
        [0, 0, 1, 1]
    )

    predictions = np.array(
        [0, 0, 1, 1]
    )

    score = cohen_kappa(
        targets,
        predictions,
    )

    assert score == 1.0


def test_evaluate_binary_classification():
    targets = torch.tensor(
        [0, 0, 1, 1]
    )

    probabilities = torch.tensor(
        [0.1, 0.2, 0.8, 0.9]
    )

    results = evaluate_binary_classification(
        targets,
        probabilities,
    )

    expected_keys = {
        "AUROC",
        "Macro_F1",
        "Balanced_Accuracy",
        "False_Positive_Rate",
        "Cohen_Kappa",
    }

    assert set(results.keys()) == expected_keys

    assert results["AUROC"] == 1.0
    assert results["Macro_F1"] == 1.0
    assert results["Balanced_Accuracy"] == 1.0
    assert results["False_Positive_Rate"] == 0.0
    assert results["Cohen_Kappa"] == 1.0


def test_auroc_single_class_returns_nan():
    targets = np.array(
        [0, 0, 0]
    )

    probabilities = np.array(
        [0.1, 0.2, 0.3]
    )

    score = auroc(
        targets,
        probabilities,
    )

    assert math.isnan(score)