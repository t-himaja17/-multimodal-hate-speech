"""
Projection and modality alignment for multimodal fusion.

Member 4 ownership:
    src/multimodal_hate/models/fusion/

Methodology:
    Image features: [B, N_img, 1024] -> [B, N_img, 512]
    Text features:  [B, N_txt, 768]  -> [B, N_txt, 512]

Each modality is processed using:
    Linear projection
    +
    LayerNorm
    +
    learnable modality-type embedding
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ModalityProjection(nn.Module):
    """
    Projects one modality into the shared fusion space.

    Parameters
    ----------
    input_dim:
        Input feature dimension.
    fusion_dim:
        Shared fusion dimension.
    """

    def __init__(self, input_dim: int, fusion_dim: int = 512) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than 0.")

        if fusion_dim <= 0:
            raise ValueError("fusion_dim must be greater than 0.")

        self.input_dim = input_dim
        self.fusion_dim = fusion_dim

        self.projection = nn.Linear(input_dim, fusion_dim)
        self.layer_norm = nn.LayerNorm(fusion_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Project input features into the shared fusion dimension.

        Parameters
        ----------
        x:
            Tensor of shape [B, N, input_dim].

        Returns
        -------
        Tensor:
            Tensor of shape [B, N, fusion_dim].
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3D tensor [B, N, D], got shape {tuple(x.shape)}."
            )

        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {x.size(-1)}."
            )

        return self.layer_norm(self.projection(x))


class ProjectionAlignment(nn.Module):
    """
    Projects image and text representations into a shared fusion space.

    Image:
        [B, N_img, image_input_dim]
        ->
        [B, N_img, fusion_dim]

    Text:
        [B, N_txt, text_input_dim]
        ->
        [B, N_txt, fusion_dim]

    Learnable modality embeddings are added after projection.
    """

    def __init__(
        self,
        image_input_dim: int = 1024,
        text_input_dim: int = 768,
        fusion_dim: int = 512,
    ) -> None:
        super().__init__()

        if image_input_dim <= 0:
            raise ValueError("image_input_dim must be greater than 0.")

        if text_input_dim <= 0:
            raise ValueError("text_input_dim must be greater than 0.")

        if fusion_dim <= 0:
            raise ValueError("fusion_dim must be greater than 0.")

        self.image_projection = ModalityProjection(
            input_dim=image_input_dim,
            fusion_dim=fusion_dim,
        )

        self.text_projection = ModalityProjection(
            input_dim=text_input_dim,
            fusion_dim=fusion_dim,
        )

        # Learnable modality-type embeddings.
        #
        # Shape:
        # [1, 1, fusion_dim]
        #
        # Broadcasting automatically applies the same modality
        # embedding to every token in every sample.
        self.img_token = nn.Parameter(
            torch.zeros(1, 1, fusion_dim)
        )

        self.txt_token = nn.Parameter(
            torch.zeros(1, 1, fusion_dim)
        )

        self.image_input_dim = image_input_dim
        self.text_input_dim = text_input_dim
        self.fusion_dim = fusion_dim

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize modality embeddings.

        Zero initialization is used so that the model initially
        relies on the projected representations while learning
        modality-specific offsets during training.
        """

        nn.init.zeros_(self.img_token)
        nn.init.zeros_(self.txt_token)

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Project and align image and text features.

        Parameters
        ----------
        image_features:
            Tensor of shape [B, N_img, image_input_dim].

        text_features:
            Tensor of shape [B, N_txt, text_input_dim].

        Returns
        -------
        projected_image:
            Tensor of shape [B, N_img, fusion_dim].

        projected_text:
            Tensor of shape [B, N_txt, fusion_dim].
        """

        if image_features.ndim != 3:
            raise ValueError(
                "image_features must have shape [B, N_img, D]. "
                f"Got {tuple(image_features.shape)}."
            )

        if text_features.ndim != 3:
            raise ValueError(
                "text_features must have shape [B, N_txt, D]. "
                f"Got {tuple(text_features.shape)}."
            )

        if image_features.size(0) != text_features.size(0):
            raise ValueError(
                "Image and text batch sizes must match. "
                f"Got {image_features.size(0)} and "
                f"{text_features.size(0)}."
            )

        projected_image = self.image_projection(image_features)
        projected_text = self.text_projection(text_features)

        projected_image = projected_image + self.img_token
        projected_text = projected_text + self.txt_token

        return projected_image, projected_text