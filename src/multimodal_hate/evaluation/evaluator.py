"""
Evaluation pipeline for the multimodal hate-speech model.

Member 4 ownership.

The evaluator:
    1. Runs the trained model on an evaluation DataLoader.
    2. Collects hate/sarcasm/target predictions.
    3. Converts logits to probabilities.
    4. Computes configured classification metrics.
    5. Returns predictions and metrics for downstream reporting.

Primary metrics:
    - AUROC
    - Macro_F1
    - Balanced_Accuracy
    - False_Positive_Rate
    - Cohen_Kappa
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from multimodal_hate.evaluation.metrics import (
    evaluate_binary_classification,
)


class MultimodalEvaluator:
    """
    Evaluate a trained MultimodalHateSpeechModel.

    Parameters
    ----------
    model:
        Trained multimodal hate-speech model.

    device:
        Evaluation device. If omitted, CUDA is used when available.

    threshold:
        Classification threshold for binary predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str | None = None,
        threshold: float = 0.5,
    ) -> None:
        if model is None:
            raise ValueError("model must not be None.")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        self.model = model

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)
        self.threshold = threshold

        self.model.to(self.device)

    # ============================================================
    # OUTPUT EXTRACTION
    # ============================================================

    @staticmethod
    def _extract_logits(
        output: Any,
    ) -> dict[str, Tensor]:
        """
        Extract classification logits from model output.

        Expected keys:

            hate
            sarcasm
            target
        """

        if hasattr(output, "logits"):
            logits = output.logits

        elif isinstance(output, dict):
            logits = output.get("logits")

        else:
            raise TypeError(
                "Model output must contain classification logits."
            )

        if not isinstance(logits, dict):
            raise TypeError(
                "Model logits must be a dictionary."
            )

        required = {
            "hate",
            "sarcasm",
            "target",
        }

        missing = required.difference(logits.keys())

        if missing:
            raise ValueError(
                "Model output is missing classification logits: "
                f"{sorted(missing)}"
            )

        return logits

    @staticmethod
    def _binary_probability(
        logits: Tensor,
    ) -> Tensor:
        """
        Convert two-class logits into positive-class probabilities.

        Input:
            [B, 2]

        Output:
            [B]
        """

        if not isinstance(logits, Tensor):
            raise TypeError(
                "Binary logits must be a torch.Tensor."
            )

        if logits.ndim != 2:
            raise ValueError(
                "Binary logits must have shape [B, 2]."
            )

        if logits.size(1) != 2:
            raise ValueError(
                "Binary logits must contain exactly two classes."
            )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        return probabilities[:, 1]

    @staticmethod
    def _target_probabilities(
        logits: Tensor,
    ) -> Tensor:
        """
        Convert target-group logits into multi-label probabilities.

        Input:
            [B, 5]

        Output:
            [B, 5]
        """

        if not isinstance(logits, Tensor):
            raise TypeError(
                "Target logits must be a torch.Tensor."
            )

        if logits.ndim != 2:
            raise ValueError(
                "Target logits must have shape [B, 5]."
            )

        if logits.size(1) != 5:
            raise ValueError(
                "Target logits must contain five target groups."
            )

        return torch.sigmoid(logits)

    # ============================================================
    # BATCH VALIDATION
    # ============================================================

    @staticmethod
    def _move_images(
        images: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Move image tensors to the evaluation device."""

        if not isinstance(images, Tensor):
            raise TypeError(
                "Batch image field must be a torch.Tensor."
            )

        return images.to(device)

    @staticmethod
    def _validate_targets(
        batch: dict[str, Any],
    ) -> None:
        """Validate required target fields."""

        required = {
            "hate_target",
            "sarcasm_target",
            "target_target",
            "text",
            "image",
        }

        missing = required.difference(batch.keys())

        if missing:
            raise ValueError(
                "Evaluation batch is missing fields: "
                f"{sorted(missing)}"
            )

    # ============================================================
    # EVALUATION
    # ============================================================

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, Any]:
        """
        Evaluate the model over a DataLoader.

        Returns
        -------
        dict
            Contains:

                metrics:
                    Classification metrics.

                predictions:
                    Raw probabilities.

                targets:
                    Ground-truth labels.

        The target-group metrics are not forced into the binary
        metric helper because target groups are multi-label.
        """

        if dataloader is None:
            raise ValueError(
                "dataloader must not be None."
            )

        self.model.eval()

        hate_probabilities = []
        sarcasm_probabilities = []
        target_probabilities = []

        hate_targets = []
        sarcasm_targets = []
        target_targets = []

        for batch in dataloader:

            if not isinstance(batch, dict):
                raise TypeError(
                    "Each DataLoader batch must be a dictionary."
                )

            self._validate_targets(batch)

            images = self._move_images(
                batch["image"],
                self.device,
            )

            texts = batch["text"]

            output = self.model(
                images,
                texts,
            )

            logits = self._extract_logits(
                output
            )

            hate_probability = self._binary_probability(
                logits["hate"]
            )

            sarcasm_probability = self._binary_probability(
                logits["sarcasm"]
            )

            target_probability = self._target_probabilities(
                logits["target"]
            )

            hate_probabilities.append(
                hate_probability.detach().cpu()
            )

            sarcasm_probabilities.append(
                sarcasm_probability.detach().cpu()
            )

            target_probabilities.append(
                target_probability.detach().cpu()
            )

            hate_targets.append(
                batch["hate_target"][:, 1]
                .detach()
                .cpu()
            )

            sarcasm_targets.append(
                batch["sarcasm_target"][:, 1]
                .detach()
                .cpu()
            )

            target_targets.append(
                batch["target_target"]
                .detach()
                .cpu()
            )

        if not hate_probabilities:
            raise ValueError(
                "Evaluation DataLoader produced no batches."
            )

        hate_probabilities = torch.cat(
            hate_probabilities,
            dim=0,
        )

        sarcasm_probabilities = torch.cat(
            sarcasm_probabilities,
            dim=0,
        )

        target_probabilities = torch.cat(
            target_probabilities,
            dim=0,
        )

        hate_targets = torch.cat(
            hate_targets,
            dim=0,
        )

        sarcasm_targets = torch.cat(
            sarcasm_targets,
            dim=0,
        )

        target_targets = torch.cat(
            target_targets,
            dim=0,
        )

        hate_metrics = evaluate_binary_classification(
            hate_targets,
            hate_probabilities,
            threshold=self.threshold,
        )

        sarcasm_metrics = evaluate_binary_classification(
            sarcasm_targets,
            sarcasm_probabilities,
            threshold=self.threshold,
        )

        return {
            "metrics": {
                "hate": hate_metrics,
                "sarcasm": sarcasm_metrics,
            },
            "predictions": {
                "hate": hate_probabilities,
                "sarcasm": sarcasm_probabilities,
                "target": target_probabilities,
            },
            "targets": {
                "hate": hate_targets,
                "sarcasm": sarcasm_targets,
                "target": target_targets,
            },
        }