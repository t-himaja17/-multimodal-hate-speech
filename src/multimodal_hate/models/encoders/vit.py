import torch

import open_clip

from .base import BaseEncoder


class ViTEncoder(BaseEncoder):
    """OpenAI CLIP ViT-L/14 image token encoder."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
    ):
        super().__init__(output_dim=1024)

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
        )

        self.visual = self.model.visual

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using OpenAI CLIP ViT-L/14.

        Expected input:
            [B, 3, 224, 224]

        Expected output:
            [B, N_img, 1024]
        """

        device = next(self.visual.parameters()).device
        images = images.to(device)

        x = self.visual.conv1(images)

        batch_size, hidden_size, grid_h, grid_w = x.shape

        x = x.reshape(
            batch_size,
            hidden_size,
            grid_h * grid_w,
        )

        x = x.permute(0, 2, 1)

        class_embedding = self.visual.class_embedding.to(x.dtype)
        class_embedding = class_embedding.expand(
            batch_size,
            1,
            -1,
        )

        x = torch.cat([class_embedding, x], dim=1)

        positional_embedding = self.visual.positional_embedding.to(x.dtype)
        x = x + positional_embedding

        x = self.visual.patch_dropout(x)

        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)

        x = self.visual.transformer(x)

        x = x.permute(1, 0, 2)

        x = self.visual.ln_post(x)

        return x


def encode_image_vit(images: torch.Tensor) -> torch.Tensor:
    """Encode images using OpenAI CLIP ViT-L/14."""

    encoder = ViTEncoder()

    return encoder(images)