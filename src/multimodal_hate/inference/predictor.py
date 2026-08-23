"""
Inference utilities for the multimodal hate-speech model.

Member 4 ownership.

Pipeline:

    image + text
        ↓
    MultimodalHateSpeechModel
        ↓
    classification logits
        ↓
    probabilities
        ↓
    structured prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image


@dataclass
class HateSpeechPrediction:
    """Structured prediction returned by the inference pipeline."""

    hate_probability: float
    hate_label: int

    sarcasm_probability: float
    sarcasm_label: int

    target_probabilities: list[float]
    target_labels: list[int]

    sample_id: Optional[str] = None


class MultimodalPredictor:
    """
    Inference wrapper around MultimodalHateSpeechModel.

    The predictor is intentionally separated from the model itself.
    The model performs neural computation, while this class handles:

        image loading
        preprocessing
        model inference
        probability conversion
        label decoding
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
        model,
        device: str | torch.device | None = None,
        image_size: int = 224,
        target_threshold: float = 0.5,
    ) -> None:
        if model is None:
            raise ValueError("model is required.")

        if image_size <= 0:
            raise ValueError(
                "image_size must be greater than zero."
            )

        if not 0.0 <= target_threshold <= 1.0:
            raise ValueError(
                "target_threshold must be between 0 and 1."
            )

        self.model = model
        self.image_size = image_size
        self.target_threshold = target_threshold

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    # ============================================================
    # IMAGE PREPROCESSING
    # ============================================================

    def load_image(
        self,
        image_path: str | Path,
    ) -> torch.Tensor:
        """
        Load and preprocess an image.

        Returns:

            [3, 224, 224]
        """

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {path}"
            )

        try:
            image = Image.open(path).convert("RGB")
        except OSError as exc:
            raise ValueError(
                f"Unable to read image: {path}"
            ) from exc

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BICUBIC,
        )

        # Avoid adding numpy as an inference dependency.
        image_tensor = torch.tensor(
            list(image.getdata()),
            dtype=torch.float32,
        )

        image_tensor = image_tensor.reshape(
            self.image_size,
            self.image_size,
            3,
        )

        image_tensor = image_tensor.permute(
            2,
            0,
            1,
        )

        image_tensor = image_tensor / 255.0

        return image_tensor

    # ============================================================
    # LOGIT VALIDATION
    # ============================================================

    @staticmethod
    def _validate_logits(
        logits: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        """Validate classification-head outputs."""

        required = (
            "hate",
            "sarcasm",
            "target",
        )

        for name in required:
            if name not in logits:
                raise ValueError(
                    f"Missing '{name}' logits."
                )

            if not isinstance(
                logits[name],
                torch.Tensor,
            ):
                raise TypeError(
                    f"{name} logits must be a tensor."
                )

        if logits["hate"].shape != (
            batch_size,
            2,
        ):
            raise ValueError(
                "Hate logits must have shape [B, 2]."
            )

        if logits["sarcasm"].shape != (
            batch_size,
            2,
        ):
            raise ValueError(
                "Sarcasm logits must have shape [B, 2]."
            )

        if logits["target"].shape != (
            batch_size,
            5,
        ):
            raise ValueError(
                "Target logits must have shape [B, 5]."
            )

    # ============================================================
    # SINGLE PREDICTION
    # ============================================================

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        text: str,
        sample_id: Optional[str] = None,
    ) -> HateSpeechPrediction:
        """
        Run inference for one image-text pair.

        Parameters
        ----------
        image:
            Image tensor [3, H, W].

        text:
            OCR/text content.

        sample_id:
            Optional identifier.

        Returns
        -------
        HateSpeechPrediction
        """

        if not isinstance(
            image,
            torch.Tensor,
        ):
            raise TypeError(
                "image must be a torch.Tensor."
            )

        if image.ndim != 3:
            raise ValueError(
                "image must have shape [3, H, W]."
            )

        if image.size(0) != 3:
            raise ValueError(
                "image must have three channels."
            )

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )

        image = image.unsqueeze(0).to(
            self.device
        )

        output = self.model(
            images=image,
            texts=[text],
        )

        logits = output.logits

        self._validate_logits(
            logits,
            batch_size=1,
        )

        hate_probabilities = torch.softmax(
            logits["hate"],
            dim=-1,
        )

        sarcasm_probabilities = torch.softmax(
            logits["sarcasm"],
            dim=-1,
        )

        target_probabilities = torch.sigmoid(
            logits["target"]
        )

        hate_label = int(
            torch.argmax(
                hate_probabilities[0]
            ).item()
        )

        sarcasm_label = int(
            torch.argmax(
                sarcasm_probabilities[0]
            ).item()
        )

        target_probs = (
            target_probabilities[0]
            .detach()
            .cpu()
            .tolist()
        )

        target_labels = [
            int(
                probability
                >= self.target_threshold
            )
            for probability in target_probs
        ]

        return HateSpeechPrediction(
            hate_probability=float(
                hate_probabilities[0, 1].item()
            ),
            hate_label=hate_label,
            sarcasm_probability=float(
                sarcasm_probabilities[0, 1].item()
            ),
            sarcasm_label=sarcasm_label,
            target_probabilities=target_probs,
            target_labels=target_labels,
            sample_id=sample_id,
        )

    # ============================================================
    # IMAGE-PATH PREDICTION
    # ============================================================

    @torch.no_grad()
    def predict_from_path(
        self,
        image_path: str | Path,
        text: str,
        sample_id: Optional[str] = None,
    ) -> HateSpeechPrediction:
        """
        Load an image from disk and run prediction.
        """

        image = self.load_image(
            image_path
        )

        return self.predict(
            image=image,
            text=text,
            sample_id=sample_id,
        )

    # ============================================================
    # BATCH PREDICTION
    # ============================================================

    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
        texts: Sequence[str],
        sample_ids: Optional[
            Sequence[str]
        ] = None,
    ) -> list[HateSpeechPrediction]:
        """
        Run inference on a batch.

        Parameters
        ----------
        images:
            Tensor [B, 3, H, W].

        texts:
            Sequence containing B text strings.

        sample_ids:
            Optional sequence of B identifiers.
        """

        if not isinstance(
            images,
            torch.Tensor,
        ):
            raise TypeError(
                "images must be a torch.Tensor."
            )

        if images.ndim != 4:
            raise ValueError(
                "images must have shape [B, 3, H, W]."
            )

        batch_size = images.size(0)

        if len(texts) != batch_size:
            raise ValueError(
                "Number of texts must match image batch size."
            )

        if not all(
            isinstance(text, str)
            for text in texts
        ):
            raise TypeError(
                "Every text item must be a string."
            )

        if sample_ids is not None:
            if len(sample_ids) != batch_size:
                raise ValueError(
                    "Number of sample_ids must match "
                    "image batch size."
                )

        images = images.to(
            self.device
        )

        output = self.model(
            images=images,
            texts=list(texts),
        )

        logits = output.logits

        self._validate_logits(
            logits,
            batch_size=batch_size,
        )

        hate_probabilities = torch.softmax(
            logits["hate"],
            dim=-1,
        )

        sarcasm_probabilities = torch.softmax(
            logits["sarcasm"],
            dim=-1,
        )

        target_probabilities = torch.sigmoid(
            logits["target"]
        )

        predictions = []

        for index in range(batch_size):

            hate_label = int(
                torch.argmax(
                    hate_probabilities[index]
                ).item()
            )

            sarcasm_label = int(
                torch.argmax(
                    sarcasm_probabilities[index]
                ).item()
            )

            target_probs = (
                target_probabilities[index]
                .detach()
                .cpu()
                .tolist()
            )

            target_labels = [
                int(
                    probability
                    >= self.target_threshold
                )
                for probability in target_probs
            ]

            sample_id = (
                sample_ids[index]
                if sample_ids is not None
                else None
            )

            predictions.append(
                HateSpeechPrediction(
                    hate_probability=float(
                        hate_probabilities[
                            index,
                            1,
                        ].item()
                    ),
                    hate_label=hate_label,
                    sarcasm_probability=float(
                        sarcasm_probabilities[
                            index,
                            1,
                        ].item()
                    ),
                    sarcasm_label=sarcasm_label,
                    target_probabilities=target_probs,
                    target_labels=target_labels,
                    sample_id=sample_id,
                )
            )

        return predictions