"""Visual-text incongruity module.

This module combines CLIP image and text embeddings to estimate
visual-text semantic incongruity.

Methodology:
    CLIP image embedding [B, 512]
    +
    CLIP text embedding [B, 512]
        ↓
    cosine similarity
        ↓
    semantic dissimilarity = 1 - similarity
        ↓
    learned incongruity head
        ↓
    incongruity score [B, 1]

The module does not implement CLIP itself. It consumes embeddings
produced by the CLIP encoders owned by Member 2.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class VisualTextIncongruity(nn.Module):
    """Calculate visual-text semantic incongruity.

    Args:
        embedding_dim:
            Expected CLIP embedding dimension.
            Project contract specifies 512.

        hidden_dim:
            Hidden dimension of the learned incongruity head.

        eps:
            Small numerical value used by cosine normalization to
            protect against zero vectors.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 128,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than zero."
            )

        if eps <= 0:
            raise ValueError(
                "eps must be greater than zero."
            )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.eps = eps

        # The learned head receives semantic dissimilarity as a scalar.
        self.incongruity_head = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _validate_embeddings(
        self,
        image_embeddings: Tensor,
        text_embeddings: Tensor,
    ) -> None:
        """Validate image and text embedding tensors."""

        if not isinstance(image_embeddings, Tensor):
            raise TypeError(
                "image_embeddings must be a torch.Tensor."
            )

        if not isinstance(text_embeddings, Tensor):
            raise TypeError(
                "text_embeddings must be a torch.Tensor."
            )

        if image_embeddings.ndim != 2:
            raise ValueError(
                "image_embeddings must have shape [B, D]."
            )

        if text_embeddings.ndim != 2:
            raise ValueError(
                "text_embeddings must have shape [B, D]."
            )

        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError(
                "Image and text embeddings must have identical "
                "batch and feature dimensions. "
                f"Got {tuple(image_embeddings.shape)} and "
                f"{tuple(text_embeddings.shape)}."
            )

        if image_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                "Unexpected embedding dimension. "
                f"Expected {self.embedding_dim}, "
                f"got {image_embeddings.shape[1]}."
            )

        if image_embeddings.device != text_embeddings.device:
            raise ValueError(
                "Image and text embeddings must be on the same device. "
                f"Got {image_embeddings.device} and "
                f"{text_embeddings.device}."
            )

        if not (
            image_embeddings.is_floating_point()
            and text_embeddings.is_floating_point()
        ):
            raise TypeError(
                "Image and text embeddings must use floating-point dtype."
            )

    def cosine_similarity(
        self,
        image_embeddings: Tensor,
        text_embeddings: Tensor,
    ) -> Tensor:
        """Return cosine similarity for each image-text pair.

        Returns:
            Tensor with shape [B, 1].
        """

        self._validate_embeddings(
            image_embeddings,
            text_embeddings,
        )

        image_norm = F.normalize(
            image_embeddings,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        text_norm = F.normalize(
            text_embeddings,
            p=2,
            dim=-1,
            eps=self.eps,
        )

        similarity = torch.sum(
            image_norm * text_norm,
            dim=-1,
            keepdim=True,
        )

        # Numerical protection.
        similarity = similarity.clamp(
            min=-1.0,
            max=1.0,
        )

        return similarity

    def semantic_dissimilarity(
        self,
        image_embeddings: Tensor,
        text_embeddings: Tensor,
    ) -> Tensor:
        """Return semantic dissimilarity = 1 - cosine similarity.

        Returns:
            Tensor with shape [B, 1] and theoretical range [0, 2].
        """

        similarity = self.cosine_similarity(
            image_embeddings,
            text_embeddings,
        )

        dissimilarity = 1.0 - similarity

        return dissimilarity

    def forward(
        self,
        image_embeddings: Tensor,
        text_embeddings: Tensor,
    ) -> Tensor:
        """Calculate the learned incongruity score.

        Args:
            image_embeddings:
                CLIP image embeddings with shape [B, 512].

            text_embeddings:
                CLIP text embeddings with shape [B, 512].

        Returns:
            Incongruity score with shape [B, 1] and range [0, 1].
        """

        dissimilarity = self.semantic_dissimilarity(
            image_embeddings,
            text_embeddings,
        )

        logits = self.incongruity_head(
            dissimilarity
        )

        score = torch.sigmoid(logits)

        return score