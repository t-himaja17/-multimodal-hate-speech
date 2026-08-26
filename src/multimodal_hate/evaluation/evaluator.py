"""
Evaluation pipeline for the multimodal hate-speech model.

Member 4 ownership.

The evaluator:
    1. Runs the trained model on an evaluation DataLoader.
    2. Collects hate/sarcasm/target predictions.
    3. Converts logits to probabilities.
    4. Computes configured classification metrics.
    5. Ignores unavailable labels represented as [0, 0].
    6. Returns predictions and targets for downstream reporting.

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

    Missing binary labels are represented by [0, 0] and are
    excluded from the corresponding binary metrics.
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
        """Extract classification logits from model output."""

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

    # ============================================================
    # PROBABILITY CONVERSION
    # ============================================================

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

        return torch.softmax(
            logits,
            dim=-1,
        )[:, 1]

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
    # TARGET HELPERS
    # ============================================================

    @staticmethod
    def _binary_target_with_mask(
        targets: Tensor,
        name: str,
    ) -> tuple[Tensor, Tensor]:
        """
        Convert one-hot binary targets into labels and validity mask.

        Valid:
            [1, 0] -> label 0
            [0, 1] -> label 1

        Missing:
            [0, 0] -> excluded from metrics

        Returns:
            labels:
                [B] binary labels for valid samples.

            mask:
                [B] True where a valid label exists.
        """

        if not isinstance(targets, Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if targets.ndim != 2 or targets.size(1) != 2:
            raise ValueError(
                f"{name} must have shape [B, 2]. "
                f"Got {tuple(targets.shape)}."
            )

        valid = targets.sum(dim=1) > 0

        invalid = valid & (
            targets.sum(dim=1) != 1
        )

        if torch.any(invalid):
            raise ValueError(
                f"{name} contains invalid binary one-hot labels."
            )

        labels = targets[:, 1].long()

        return labels, valid

    @staticmethod
    def _target_group_mask(
        targets: Tensor,
    ) -> Tensor:
        """
        Determine which target-group rows contain annotations.

        A row containing all zeros means target-group annotation
        is unavailable.
        """

        if not isinstance(targets, Tensor):
            raise TypeError(
                "target_target must be a torch.Tensor."
            )

        if targets.ndim != 2 or targets.size(1) != 5:
            raise ValueError(
                "target_target must have shape [B, 5]. "
                f"Got {tuple(targets.shape)}."
            )

        return targets.sum(dim=1) > 0

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

        Missing binary labels represented by [0, 0] are excluded
        from the corresponding metrics.

        Target-group predictions and targets are always returned,
        but target-group metrics are not calculated here because
        they are multi-label rather than binary.
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

        hate_masks = []
        sarcasm_masks = []
        target_masks = []

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

            logits = self._extract_logits(output)

            hate_probability = self._binary_probability(
                logits["hate"]
            )

            sarcasm_probability = self._binary_probability(
                logits["sarcasm"]
            )

            target_probability = self._target_probabilities(
                logits["target"]
            )

            hate_target = batch["hate_target"].detach().cpu()
            sarcasm_target = (
                batch["sarcasm_target"].detach().cpu()
            )
            target_target = (
                batch["target_target"].detach().cpu()
            )

            _, hate_mask = self._binary_target_with_mask(
                hate_target,
                "hate_target",
            )

            _, sarcasm_mask = self._binary_target_with_mask(
                sarcasm_target,
                "sarcasm_target",
            )

            target_mask = self._target_group_mask(
                target_target
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

            hate_targets.append(hate_target)
            sarcasm_targets.append(sarcasm_target)
            target_targets.append(target_target)

            hate_masks.append(hate_mask)
            sarcasm_masks.append(sarcasm_mask)
            target_masks.append(target_mask)

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

        hate_masks = torch.cat(
            hate_masks,
            dim=0,
        )

        sarcasm_masks = torch.cat(
            sarcasm_masks,
            dim=0,
        )

        target_masks = torch.cat(
            target_masks,
            dim=0,
        )

        # --------------------------------------------------------
        # HATE METRICS
        # --------------------------------------------------------

        hate_labels = hate_targets[:, 1].long()

        if not torch.all(hate_masks):
            raise ValueError(
                "Hateful Memes evaluation requires hate labels "
                "for every sample."
            )

        hate_metrics = evaluate_binary_classification(
            hate_labels,
            hate_probabilities,
            threshold=self.threshold,
        )

        # --------------------------------------------------------
        # SARCASM METRICS
        # --------------------------------------------------------

        if torch.any(sarcasm_masks):

            sarcasm_labels = (
                sarcasm_targets[sarcasm_masks, 1]
                .long()
            )

            sarcasm_metrics = evaluate_binary_classification(
                sarcasm_labels,
                sarcasm_probabilities[sarcasm_masks],
                threshold=self.threshold,
            )

        else:
            sarcasm_metrics = None

        # --------------------------------------------------------
        # TARGET GROUP AVAILABILITY
        # --------------------------------------------------------

        target_metrics = None

        if torch.any(target_masks):
            target_metrics = {
                "available_samples": int(
                    target_masks.sum().item()
                ),
                "total_samples": int(
                    target_masks.numel()
                ),
            }

        return {
            "metrics": {
                "hate": hate_metrics,
                "sarcasm": sarcasm_metrics,
                "target": target_metrics,
            },
            "predictions": {
                "hate": hate_probabilities,
                "sarcasm": sarcasm_probabilities,
                "target": target_probabilities,
            },
            "targets": {
                "hate": hate_labels,
                "sarcasm": sarcasm_targets[:, 1].long(),
                "target": target_targets,
            },
            "label_masks": {
                "hate": hate_masks,
                "sarcasm": sarcasm_masks,
                "target": target_masks,
            },
        }