"""
PyTorch dataset for the multimodal hate-speech training pipeline.

Member 4 ownership.

Converts canonical MultimodalSample objects into tensors/batches
consumed by MultimodalHateSpeechModel.

Output sample:

    {
        "image": Tensor [3, 224, 224],
        "text": str,
        "hate_target": Tensor [2],
        "sarcasm_target": Tensor [2],
        "target_target": Tensor [5],
        "sample_id": str,
    }

Classification targets use one-hot encoding for the two-class
hate/sarcasm heads and multi-hot encoding for target groups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from multimodal_hate.data.schema import MultimodalSample


class MultimodalHateSpeechDataset(Dataset):
    """
    Dataset adapter for MultimodalSample objects.

    Parameters
    ----------
    samples:
        Sequence of canonical MultimodalSample objects.

    image_size:
        Image size expected by the ViT encoder.

    image_transform:
        Optional custom image transformation.

    target_groups:
        Ordered target-group labels used for the five-dimensional
        multi-label target vector.
    """

    DEFAULT_TARGET_GROUPS = (
        "race",
        "religion",
        "gender",
        "disability",
        "sexuality",
    )

    def __init__(
        self,
        samples: Sequence[MultimodalSample],
        image_size: int = 224,
        image_transform: Callable | None = None,
        target_groups: Sequence[str] | None = None,
    ) -> None:
        if not samples:
            raise ValueError("samples must contain at least one sample.")

        if image_size <= 0:
            raise ValueError(
                "image_size must be greater than zero."
            )

        self.samples = list(samples)
        self.image_size = image_size
        self.image_transform = image_transform

        if target_groups is None:
            target_groups = self.DEFAULT_TARGET_GROUPS

        target_groups = tuple(target_groups)

        if len(target_groups) != 5:
            raise ValueError(
                "Exactly five target groups are required."
            )

        if len(set(target_groups)) != len(target_groups):
            raise ValueError(
                "target_groups must not contain duplicates."
            )

        self.target_groups = target_groups

    def __len__(self) -> int:
        return len(self.samples)

    # ============================================================
    # IMAGE
    # ============================================================

    def _load_image(
        self,
        image_path: str | None,
    ) -> Tensor:
        """Load an RGB image and convert it to [3, H, W]."""

        if not image_path:
            raise ValueError(
                "Sample does not contain an image path."
            )

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

        if self.image_transform is not None:
            return self.image_transform(image)

        # Basic default preprocessing.
        #
        # Resize while keeping the Dataset dependency-light.
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BICUBIC,
        )

        # Convert PIL RGB image to float tensor [0, 1].
        tensor = torch.from_numpy(
            __import__("numpy").array(image)
        )

        tensor = tensor.permute(2, 0, 1).float() / 255.0

        return tensor

    # ============================================================
    # TEXT
    # ============================================================

    @staticmethod
    def _load_text(
        text: str | None,
    ) -> str:
        """Normalize missing text to an empty string."""

        if text is None:
            return ""

        return str(text)

    # ============================================================
    # BINARY TARGETS
    # ============================================================

    @staticmethod
    def _binary_target(
        label: int | None,
        name: str,
    ) -> Tensor:
        """
        Convert a binary label into a two-class one-hot vector.

        Valid labels:

            0 -> [1, 0]
            1 -> [0, 1]

        Missing labels:

            -1 -> [0, 0]

        The -1 representation allows the training pipeline to
        identify unavailable labels later.
        """

        if label is None:
            return torch.zeros(
                2,
                dtype=torch.float32,
            )

        if label not in (0, 1):
            raise ValueError(
                f"{name} must be 0, 1, or None. "
                f"Got {label!r}."
            )

        target = torch.zeros(
            2,
            dtype=torch.float32,
        )

        target[label] = 1.0

        return target

    # ============================================================
    # TARGET GROUP
    # ============================================================

    def _target_group_vector(
        self,
        sample: MultimodalSample,
    ) -> Tensor:
        """
        Convert target-group information into a five-dimensional
        multi-hot vector.

        Target group may be stored directly in target_group or in
        metadata["target_group"] by the dataset loaders.
        """

        target = torch.zeros(
            len(self.target_groups),
            dtype=torch.float32,
        )

        group = sample.target_group

        if group is None:
            group = sample.metadata.get("target_group")

        if group is None:
            return target

        if isinstance(group, str):
            groups = [group]
        elif isinstance(group, (list, tuple, set)):
            groups = list(group)
        else:
            raise TypeError(
                "target_group must be a string or sequence of strings."
            )

        group_to_index = {
            name.lower(): index
            for index, name in enumerate(self.target_groups)
        }

        for item in groups:
            if not isinstance(item, str):
                raise TypeError(
                    "Every target-group label must be a string."
                )

            normalized = item.strip().lower()

            if normalized in group_to_index:
                target[group_to_index[normalized]] = 1.0

        return target

    # ============================================================
    # GET ITEM
    # ============================================================

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        """Return one training sample."""

        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        sample = self.samples[index]

        image = self._load_image(
            sample.image_path
        )

        text = self._load_text(
            sample.text
        )

        hate_target = self._binary_target(
            sample.hate_label,
            "hate_label",
        )

        sarcasm_target = self._binary_target(
            sample.sarcasm_label,
            "sarcasm_label",
        )

        target_target = self._target_group_vector(
            sample
        )

        return {
            "image": image,
            "text": text,
            "hate_target": hate_target,
            "sarcasm_target": sarcasm_target,
            "target_target": target_target,
            "sample_id": sample.sample_id,
        }


def multimodal_collate_fn(
    batch: list[dict[str, object]],
) -> dict[str, object]:
    """
    Collate Dataset samples into a model-ready batch.

    Returns:

        image:
            [B, 3, 224, 224]

        text:
            list[str]

        hate_target:
            [B, 2]

        sarcasm_target:
            [B, 2]

        target_target:
            [B, 5]

        sample_id:
            list[str]
    """

    if not batch:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    images = torch.stack(
        [
            item["image"]
            for item in batch
        ]
    )

    hate_targets = torch.stack(
        [
            item["hate_target"]
            for item in batch
        ]
    )

    sarcasm_targets = torch.stack(
        [
            item["sarcasm_target"]
            for item in batch
        ]
    )

    target_targets = torch.stack(
        [
            item["target_target"]
            for item in batch
        ]
    )

    texts = [
        str(item["text"])
        for item in batch
    ]

    sample_ids = [
        str(item["sample_id"])
        for item in batch
    ]

    return {
        "image": images,
        "text": texts,
        "hate_target": hate_targets,
        "sarcasm_target": sarcasm_targets,
        "target_target": target_targets,
        "sample_id": sample_ids,
    }