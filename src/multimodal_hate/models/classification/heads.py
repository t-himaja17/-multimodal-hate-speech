"""
Classification heads for the multimodal hate-speech model.

Member 4 ownership.

Heads:
    - Hate speech classification: 2 classes
    - Sarcasm classification: 2 classes
    - Target-group classification: 5 labels

The heads return raw logits.
Loss functions such as BCEWithLogitsLoss are applied later
by the training/loss pipeline.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """
    Generic classification head.

    Architecture:

        input
          ↓
        LayerNorm
          ↓
        Linear
          ↓
        GELU
          ↓
        Dropout
          ↓
        Linear
          ↓
        logits
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be greater than 0."
            )

        if output_dim <= 0:
            raise ValueError(
                "output_dim must be greater than 0."
            )

        if hidden_dim is None:
            hidden_dim = input_dim

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than 0."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.norm = nn.LayerNorm(input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            Representation with shape [B, input_dim].

        Returns
        -------
        Tensor:
            Raw classification logits with shape [B, output_dim].
        """

        if x.ndim != 2:
            raise ValueError(
                "Classification input must have shape [B, D]. "
                f"Got {tuple(x.shape)}."
            )

        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {x.size(-1)}."
            )

        x = self.norm(x)

        return self.classifier(x)


class HateClassificationHead(ClassificationHead):
    """
    Binary hate-speech classification head.

    Output:
        [B, 2]
    """

    def __init__(
        self,
        input_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=2,
            dropout=dropout,
        )


class SarcasmClassificationHead(ClassificationHead):
    """
    Binary sarcasm classification head.

    Output:
        [B, 2]
    """

    def __init__(
        self,
        input_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=2,
            dropout=dropout,
        )


class TargetGroupClassificationHead(ClassificationHead):
    """
    Multi-label target-group classification head.

    Target groups:

        0 → race
        1 → religion
        2 → gender
        3 → disability
        4 → sexuality

    Output:
        [B, 5]

    The output contains raw logits. Apply sigmoid only when
    converting logits into independent target probabilities.
    """

    TARGET_GROUPS = (
        "race",
        "religion",
        "gender",
        "disability",
        "sexuality",
    )

    def __init__(
        self,
        input_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=len(self.TARGET_GROUPS),
            dropout=dropout,
        )

    @property
    def target_groups(self) -> tuple[str, ...]:
        """Return target-group labels in classifier order."""
        return self.TARGET_GROUPS


class MultitaskClassificationHeads(nn.Module):
    """
    Complete classification-head container.

    Inputs:
        pooled fused representation [B, 512]

    Outputs:
        hate_logits    [B, 2]
        sarcasm_logits [B, 2]
        target_logits  [B, 5]
    """

    def __init__(
        self,
        input_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.dropout = dropout

        self.hate = HateClassificationHead(
            input_dim=input_dim,
            dropout=dropout,
        )

        self.sarcasm = SarcasmClassificationHead(
            input_dim=input_dim,
            dropout=dropout,
        )

        self.target = TargetGroupClassificationHead(
            input_dim=input_dim,
            dropout=dropout,
        )

    def forward(
        self,
        pooled_representation: Tensor,
    ) -> dict[str, Tensor]:
        """
        Generate all classification logits.

        Parameters
        ----------
        pooled_representation:
            [B, input_dim]

        Returns
        -------
        dict:
            {
                "hate": [B, 2],
                "sarcasm": [B, 2],
                "target": [B, 5],
            }
        """

        if pooled_representation.ndim != 2:
            raise ValueError(
                "pooled_representation must have shape [B, D]. "
                f"Got {tuple(pooled_representation.shape)}."
            )

        if pooled_representation.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected pooled representation dimension "
                f"{self.input_dim}, got "
                f"{pooled_representation.size(-1)}."
            )

        return {
            "hate": self.hate(pooled_representation),
            "sarcasm": self.sarcasm(pooled_representation),
            "target": self.target(pooled_representation),
        }