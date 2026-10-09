from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class BinaryClassificationLoss(nn.Module):
    """
    Two-class cross-entropy loss for one-hot binary targets.

    Label encoding:
        0 -> [1, 0]
        1 -> [0, 1]
        None -> [0, 0]

    Missing-label samples are ignored.
    positive_weight applies only to class 1.
    """

    def __init__(
        self,
        positive_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if positive_weight <= 0:
            raise ValueError(
                "positive_weight must be greater than zero."
            )

        self.positive_weight = float(positive_weight)

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

        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(
                "BinaryClassificationLoss expects logits "
                "and one-hot targets with shape [batch_size, 2]."
            )

        targets = targets.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        # [0, 0] denotes a missing label.
        valid = targets.sum(dim=1) > 0

        if not torch.any(valid):
            return logits.sum() * 0.0

        # Convert [1, 0] -> 0 and [0, 1] -> 1.
        labels = targets.argmax(dim=1).long()

        # Weight class 1 only; class 0 remains weight 1.
        class_weights = torch.tensor(
            [1.0, self.positive_weight],
            device=logits.device,
            dtype=logits.dtype,
        )

        per_sample_loss = F.cross_entropy(
            logits[valid],
            labels[valid],
            weight=class_weights,
            reduction="none",
        )

        return per_sample_loss.mean()


class MultiLabelBCELoss(nn.Module):
    """
    Multi-label BCE loss for target groups.

    Samples with no target annotations are ignored.
    """

    def __init__(
        self,
        positive_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if positive_weight <= 0:
            raise ValueError(
                "positive_weight must be greater than zero."
            )

        self.register_buffer(
            "positive_weight",
            torch.tensor(float(positive_weight)),
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

        if logits.ndim != 2:
            raise ValueError(
                "MultiLabelBCELoss expects [batch_size, num_labels]."
            )

        targets = targets.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        # Preserve the project's missing-annotation convention.
        valid = targets.sum(dim=1) > 0

        if not torch.any(valid):
            return logits.sum() * 0.0

        positive_weight = self.positive_weight.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        return F.binary_cross_entropy_with_logits(
            logits[valid],
            targets[valid],
            pos_weight=positive_weight,
            reduction="mean",
        )


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy.

    Same-index image/text representations are positive pairs.
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

        representation_a = F.normalize(
            representation_a,
            dim=-1,
        )

        representation_b = F.normalize(
            representation_b,
            dim=-1,
        )

        representations = torch.cat(
            [representation_a, representation_b],
            dim=0,
        )

        similarity = torch.matmul(
            representations,
            representations.T,
        ) / self.temperature

        mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=similarity.device,
        )

        similarity = similarity.masked_fill(
            mask,
            torch.finfo(similarity.dtype).min,
        )

        positive_indices = (
            torch.arange(
                2 * batch_size,
                device=similarity.device,
            )
            + batch_size
        ) % (2 * batch_size)

        return F.cross_entropy(
            similarity,
            positive_indices,
        )


class MultimodalTotalLoss(nn.Module):
    """
    Complete weighted multimodal training loss.

    Default weights:
        hate        = 1.0
        sarcasm     = 0.3
        contrastive = 0.1
        target      = 0.2

    sarcasm_positive_weight weights class 1 (sarcasm)
    without increasing the weight of class 0.
    """

    def __init__(
        self,
        hate_weight: float = 1.0,
        sarcasm_weight: float = 0.3,
        contrastive_weight: float = 0.1,
        target_weight: float = 0.2,
        contrastive_temperature: float = 0.07,
        sarcasm_positive_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if hate_weight < 0:
            raise ValueError(
                "hate_weight must be non-negative."
            )

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

        if sarcasm_positive_weight <= 0:
            raise ValueError(
                "sarcasm_positive_weight must be greater than zero."
            )

        self.hate_weight = hate_weight
        self.sarcasm_weight = sarcasm_weight
        self.contrastive_weight = contrastive_weight
        self.target_weight = target_weight

        self.hate_loss = BinaryClassificationLoss()

        self.sarcasm_loss = BinaryClassificationLoss(
            positive_weight=sarcasm_positive_weight,
        )

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
    ) -> Dict[str, Tensor]:

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
            self.hate_weight * hate
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