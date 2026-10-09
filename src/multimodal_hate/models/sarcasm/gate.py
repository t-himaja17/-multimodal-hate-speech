"""Sarcasm gate for sarcasm-conditioned multimodal fusion.

Methodology:

    sarcasm probability [B, 1]
             +
    incongruity score [B, 1]
             +
    sentiment reversal [B, 1]
             |
             v
        concatenation [B, 3]
             |
             v
            MLP
             |
             v
          sigmoid
             |
             v
       sarcasm gate [B, d]

The gate dimension ``d`` is configurable because the research
methodology does not currently specify a final value.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class SarcasmGate(nn.Module):
    """Combine sarcasm signals into a learnable gate.

    Args:
        gate_dim:
            Output dimension ``d`` of the sarcasm gate.
            The research specification leaves this dimension TBD,
            therefore it is configurable.

        hidden_dim:
            Hidden dimension of the MLP.

        dropout:
            Dropout probability used between MLP layers.

    Inputs:
        sarcasm_probability:
            Tensor of shape [B, 1].

        incongruity_score:
            Tensor of shape [B, 1].

        sentiment_reversal:
            Tensor of shape [B, 1].

    Output:
        Tensor of shape [B, gate_dim], with values in [0, 1].
    """

    def __init__(
        self,
        gate_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(gate_dim, int):
            raise TypeError("gate_dim must be an integer.")

        if gate_dim <= 0:
            raise ValueError(
                "gate_dim must be greater than zero."
            )

        if hidden_dim is None:
            hidden_dim = gate_dim

        if not isinstance(hidden_dim, int):
            raise TypeError(
                "hidden_dim must be an integer."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        self.gate_dim = gate_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gate_dim),
        )

        self.activation = nn.Sigmoid()

    @staticmethod
    def _validate_signal(
        signal: Tensor,
        name: str,
    ) -> None:
        """Validate one sarcasm signal."""

        if not isinstance(signal, Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if signal.ndim != 2:
            raise ValueError(
                f"{name} must have shape [B, 1], "
                f"got {tuple(signal.shape)}."
            )

        if signal.shape[1] != 1:
            raise ValueError(
                f"{name} must have shape [B, 1], "
                f"got {tuple(signal.shape)}."
            )

        if not torch.is_floating_point(signal):
            raise TypeError(
                f"{name} must be a floating-point tensor."
            )

    @staticmethod
    def _validate_batch_sizes(
        sarcasm_probability: Tensor,
        incongruity_score: Tensor,
        sentiment_reversal: Tensor,
    ) -> None:
        """Ensure all signals have the same batch size."""

        batch_sizes = {
            sarcasm_probability.shape[0],
            incongruity_score.shape[0],
            sentiment_reversal.shape[0],
        }

        if len(batch_sizes) != 1:
            raise ValueError(
                "All sarcasm signals must have the same batch size."
            )

    @staticmethod
    def _validate_devices(
        sarcasm_probability: Tensor,
        incongruity_score: Tensor,
        sentiment_reversal: Tensor,
    ) -> None:
        """Ensure all signals are on the same device."""

        devices = {
            sarcasm_probability.device,
            incongruity_score.device,
            sentiment_reversal.device,
        }

        if len(devices) != 1:
            raise ValueError(
                "All sarcasm signals must be on the same device."
            )

    def forward(
        self,
        sarcasm_probability: Tensor,
        incongruity_score: Tensor,
        sentiment_reversal: Tensor,
    ) -> Tensor:
        """Compute the sarcasm gate.

        Returns:
            Tensor with shape [B, gate_dim] and values in [0, 1].
        """

        self._validate_signal(
            sarcasm_probability,
            "sarcasm_probability",
        )

        self._validate_signal(
            incongruity_score,
            "incongruity_score",
        )

        self._validate_signal(
            sentiment_reversal,
            "sentiment_reversal",
        )

        self._validate_batch_sizes(
            sarcasm_probability,
            incongruity_score,
            sentiment_reversal,
        )

        self._validate_devices(
            sarcasm_probability,
            incongruity_score,
            sentiment_reversal,
        )

        if (
            sarcasm_probability.shape[0] == 0
        ):
            raise ValueError(
                "The batch dimension cannot be zero."
            )

        features = torch.cat(
            [
                sarcasm_probability,
                incongruity_score,
                sentiment_reversal,
            ],
            dim=1,
        )

        # features: [B, 3]
        logits = self.mlp(features)

        # gate: [B, d]
        gate = self.activation(logits)

        return gate