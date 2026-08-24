"""
Loss functions for the multimodal hate-speech model.

Member 4 ownership.

Configured losses:
    - Hate classification: BCE
    - Sarcasm classification: BCE
    - Target groups: Multi-label BCE
    - Contrastive alignment: NT-Xent

Total loss:

    L_total =
        L_hate
        + 0.3 * L_sarcasm
        + 0.1 * L_contrastive
        + 0.2 * L_target
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BinaryClassificationLoss(nn.Module):
    """
    Binary cross-entropy loss operating on one-hot targets.

    Valid binary labels:

        [1, 0] -> class 0
        [0, 1] -> class 1

    Missing labels:

        [0, 0]

    Missing labels are ignored instead of being treated as
    negative examples.

    BCEWithLogitsLoss is used so sigmoid and BCE are combined
    into one numerically stable operation.
    """

    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss(
            reduction="none"
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                "logits and targets must have the same shape. "
                f"Got {tuple(logits.shape)} and "
                f"{tuple(targets.shape)}."
            )

        targets = targets.float()

        # A valid one-hot binary label has exactly one positive
        # entry. [0, 0] means that the label is unavailable.
        valid = targets.sum(dim=1) > 0

        # If this task has no labels in the current batch,
        # return zero while preserving the computation graph.
        if not torch.any(valid):
            return logits.sum() * 0.0

        elementwise_loss = self.loss(
            logits,
            targets,
        )

        valid_loss = elementwise_loss[valid]

        return valid_loss.mean()


class MultiLabelBCELoss(BinaryClassificationLoss):
    """
    Multi-label BCE loss.

    Used for the five target groups:

        race
        religion
        gender
        disability
        sexuality

    A sample with no target-group annotations is represented
    by an all-zero target vector and is ignored.
    """

    pass


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy loss.

    This implementation aligns two modality representations.

    Given:

        z_a = modality A representations
        z_b = modality B representations

    samples at the same batch index are treated as positive
    pairs, while the remaining batch samples are negatives.
    """

    def __init__(
        self,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than 0."
            )

        self.temperature = temperature

    def forward(
        self,
        representation_a: Tensor,
        representation_b: Tensor,
    ) -> Tensor:
        if representation_a.ndim != 2:
            raise ValueError(
                "representation_a must have shape [B, D]. "
                f"Got {tuple(representation_a.shape)}."
            )

        if representation_b.ndim != 2:
            raise ValueError(
                "representation_b must have shape [B, D]. "
                f"Got {tuple(representation_b.shape)}."
            )

        if representation_a.shape != representation_b.shape:
            raise ValueError(
                "Both representations must have the same shape. "
                f"Got {tuple(representation_a.shape)} and "
                f"{tuple(representation_b.shape)}."
            )

        batch_size = representation_a.size(0)

        if batch_size < 2:
            raise ValueError(
                "NT-Xent requires a batch size of at least 2."
            )

        z_a = F.normalize(
            representation_a,
            dim=-1,
        )

        z_b = F.normalize(
            representation_b,
            dim=-1,
        )

        representations = torch.cat(
            [
                z_a,
                z_b,
            ],
            dim=0,
        )

        similarity = torch.matmul(
            representations,
            representations.transpose(0, 1),
        )

        similarity = similarity / self.temperature

        total_samples = 2 * batch_size

        mask = torch.eye(
            total_samples,
            dtype=torch.bool,
            device=similarity.device,
        )

        similarity = similarity.masked_fill(
            mask,
            float("-inf"),
        )

        positive_indices = torch.cat(
            [
                torch.arange(
                    batch_size,
                    total_samples,
                    device=similarity.device,
                ),
                torch.arange(
                    0,
                    batch_size,
                    device=similarity.device,
                ),
            ]
        )

        return F.cross_entropy(
            similarity,
            positive_indices,
        )


class MultimodalTotalLoss(nn.Module):
    """
    Complete weighted multimodal training loss.

    Configuration:

        hate        = BCE
        sarcasm     = BCE
        contrastive = NT-Xent
        target      = Multi-label BCE

    Weights:

        hate        = 1.0
        sarcasm     = 0.3
        contrastive = 0.1
        target      = 0.2
    """

    def __init__(
        self,
        sarcasm_weight: float = 0.3,
        contrastive_weight: float = 0.1,
        target_weight: float = 0.2,
        contrastive_temperature: float = 0.07,
    ) -> None:
        super().__init__()

        if sarcasm_weight < 0:
            raise ValueError(
                "sarcasm_weight must be non-negative."
            )

        if contrastive_weight < 0:
            raise ValueError(
                "contrastive_weight must be non-negative."
            )

        if target_weight < 0:
            raise ValueError(
                "target_weight must be non-negative."
            )

        self.sarcasm_weight = sarcasm_weight
        self.contrastive_weight = contrastive_weight
        self.target_weight = target_weight

        self.hate_loss = BinaryClassificationLoss()

        self.sarcasm_loss = BinaryClassificationLoss()

        self.target_loss = MultiLabelBCELoss()

        self.contrastive_loss = NTXentLoss(
            temperature=contrastive_temperature,
        )

    def forward(
        self,
        hate_logits: Tensor,
        hate_targets: Tensor,
        sarcasm_logits: Tensor,
        sarcasm_targets: Tensor,
        target_logits: Tensor,
        target_targets: Tensor,
        image_representation: Tensor,
        text_representation: Tensor,
    ) -> dict[str, Tensor]:
        """
        Calculate all component losses and the weighted total.

        Returns:

            {
                "hate": ...,
                "sarcasm": ...,
                "contrastive": ...,
                "target": ...,
                "total": ...
            }
        """

        hate = self.hate_loss(
            hate_logits,
            hate_targets,
        )

        sarcasm = self.sarcasm_loss(
            sarcasm_logits,
            sarcasm_targets,
        )

        target = self.target_loss(
            target_logits,
            target_targets,
        )

        contrastive = self.contrastive_loss(
            image_representation,
            text_representation,
        )

        total = (
            hate
            + self.sarcasm_weight * sarcasm
            + self.contrastive_weight * contrastive
            + self.target_weight * target
        )

        return {
            "hate": hate,
            "sarcasm": sarcasm,
            "contrastive": contrastive,
            "target": target,
            "total": total,
        }