"""
Evaluation metrics for the multimodal hate-speech model.

Member 4 ownership.

Metrics:
    - AUROC
    - Macro F1
    - Balanced Accuracy
    - False Positive Rate
    - Cohen's Kappa

The functions operate on predictions and targets and are
independent of the model architecture.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


def _to_numpy(value) -> np.ndarray:
    """Convert tensors or array-like values to NumPy arrays."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()

    return np.asarray(value)


def _validate_binary_inputs(
    targets,
    probabilities,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary target/probability inputs."""

    targets = _to_numpy(targets)
    probabilities = _to_numpy(probabilities)

    targets = targets.reshape(-1)
    probabilities = probabilities.reshape(-1)

    if targets.shape != probabilities.shape:
        raise ValueError(
            "targets and probabilities must have the same shape. "
            f"Got {targets.shape} and {probabilities.shape}."
        )

    if targets.size == 0:
        raise ValueError("targets cannot be empty.")

    return targets.astype(int), probabilities.astype(float)


def binary_predictions(
    probabilities,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert binary probabilities into 0/1 predictions."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = _to_numpy(probabilities).reshape(-1)

    return (probabilities >= threshold).astype(int)


def auroc(
    targets,
    probabilities,
) -> float:
    """
    Calculate binary AUROC.

    Returns NaN when only one target class is present because
    AUROC is undefined in that situation.
    """

    targets, probabilities = _validate_binary_inputs(
        targets,
        probabilities,
    )

    if np.unique(targets).size < 2:
        return float("nan")

    return float(
        roc_auc_score(
            targets,
            probabilities,
        )
    )


def macro_f1(
    targets,
    predictions,
) -> float:
    """Calculate binary Macro-F1."""

    targets = _to_numpy(targets).reshape(-1).astype(int)
    predictions = _to_numpy(predictions).reshape(-1).astype(int)

    if targets.shape != predictions.shape:
        raise ValueError(
            "targets and predictions must have the same shape."
        )

    if targets.size == 0:
        raise ValueError("targets cannot be empty.")

    return float(
        f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )


def balanced_accuracy(
    targets,
    predictions,
) -> float:
    """Calculate balanced accuracy."""

    targets = _to_numpy(targets).reshape(-1).astype(int)
    predictions = _to_numpy(predictions).reshape(-1).astype(int)

    if targets.shape != predictions.shape:
        raise ValueError(
            "targets and predictions must have the same shape."
        )

    if targets.size == 0:
        raise ValueError("targets cannot be empty.")

    return float(
        balanced_accuracy_score(
            targets,
            predictions,
        )
    )


def false_positive_rate(
    targets,
    predictions,
) -> float:
    """
    Calculate binary false-positive rate.

    FPR = FP / (FP + TN)

    Returns NaN when there are no negative samples.
    """

    targets = _to_numpy(targets).reshape(-1).astype(int)
    predictions = _to_numpy(predictions).reshape(-1).astype(int)

    if targets.shape != predictions.shape:
        raise ValueError(
            "targets and predictions must have the same shape."
        )

    if targets.size == 0:
        raise ValueError("targets cannot be empty.")

    negative_mask = targets == 0

    negative_count = int(
        negative_mask.sum()
    )

    if negative_count == 0:
        return float("nan")

    false_positives = int(
        ((predictions == 1) & negative_mask).sum()
    )

    return float(
        false_positives / negative_count
    )


def cohen_kappa(
    targets,
    predictions,
) -> float:
    """Calculate Cohen's Kappa."""

    targets = _to_numpy(targets).reshape(-1).astype(int)
    predictions = _to_numpy(predictions).reshape(-1).astype(int)

    if targets.shape != predictions.shape:
        raise ValueError(
            "targets and predictions must have the same shape."
        )

    if targets.size == 0:
        raise ValueError("targets cannot be empty.")

    return float(
        cohen_kappa_score(
            targets,
            predictions,
        )
    )


def evaluate_binary_classification(
    targets,
    probabilities,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calculate all configured binary classification metrics.

    Returns:

        {
            "AUROC": ...,
            "Macro_F1": ...,
            "Balanced_Accuracy": ...,
            "False_Positive_Rate": ...,
            "Cohen_Kappa": ...
        }
    """

    targets, probabilities = _validate_binary_inputs(
        targets,
        probabilities,
    )

    predictions = binary_predictions(
        probabilities,
        threshold=threshold,
    )

    return {
        "AUROC": auroc(
            targets,
            probabilities,
        ),
        "Macro_F1": macro_f1(
            targets,
            predictions,
        ),
        "Balanced_Accuracy": balanced_accuracy(
            targets,
            predictions,
        ),
        "False_Positive_Rate": false_positive_rate(
            targets,
            predictions,
        ),
        "Cohen_Kappa": cohen_kappa(
            targets,
            predictions,
        ),
    }