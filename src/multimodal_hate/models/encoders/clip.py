import torch
from transformers import CLIPModel, CLIPProcessor

from .base import BaseEncoder


class CLIPImageEncoder(BaseEncoder):
    """CLIP ViT-B/32 visual encoder."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__(output_dim=512)

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using CLIP.

        Expected input:
            [B, 3, H, W]

        Expected output:
            [B, 512]
        """

        device = next(self.model.parameters()).device
        images = images.to(device)

        outputs = self.model.get_image_features(
            pixel_values=images
        )

        # Transformers versions may return a model-output object.
        # Extract the actual image embedding tensor.
        if not isinstance(outputs, torch.Tensor):
            if hasattr(outputs, "pooler_output"):
                outputs = outputs.pooler_output
            elif hasattr(outputs, "image_embeds"):
                outputs = outputs.image_embeds
            elif hasattr(outputs, "last_hidden_state"):
                outputs = outputs.last_hidden_state[:, 0]
            else:
                raise TypeError(
                    f"Unexpected CLIP image output type: {type(outputs)}"
                )

        return outputs


def encode_image_clip(images: torch.Tensor) -> torch.Tensor:
    """Encode images using the CLIP visual encoder."""

    encoder = CLIPImageEncoder()

    return encoder(images)


class CLIPTextEncoder(BaseEncoder):
    """CLIP ViT-B/32 text encoder."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
    ):
        super().__init__(output_dim=512)

        self.max_length = max_length

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def forward(self, text) -> torch.Tensor:
        """
        Encode text using CLIP.

        Expected input:
            List of text strings.

        Expected output:
            [B, 512]
        """

        device = next(self.model.parameters()).device

        encoded = self.processor(
            text=text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Transformers versions may return a model-output object.
        # Extract the actual text embedding tensor.
        if not isinstance(outputs, torch.Tensor):
            if hasattr(outputs, "pooler_output"):
                outputs = outputs.pooler_output
            elif hasattr(outputs, "text_embeds"):
                outputs = outputs.text_embeds
            elif hasattr(outputs, "last_hidden_state"):
                outputs = outputs.last_hidden_state[:, 0]
            else:
                raise TypeError(
                    f"Unexpected CLIP text output type: {type(outputs)}"
                )

        return outputs


def encode_text_clip(text):
    """Encode text using the CLIP text encoder."""

    encoder = CLIPTextEncoder()

    return encoder(text)