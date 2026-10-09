import json
from pathlib import Path
from typing import List

from src.multimodal_hate.data.schema import MultimodalSample


def load_hateful_memes(
    annotation_file: str,
    image_dir: str,
) -> List[MultimodalSample]:
    """
    Load the Hateful Memes dataset from a JSONL annotation file.

    Supported annotation format per line:

        {
            "id": "123",
            "img": "img/123.png",
            "text": "example text",
            "label": 1,
            "sarcasm_label": 0,
            "target_group": ["gender"],
            "incongruity_type": "Hyperbole"
        }

    The original Hateful Memes fields are preserved, while optional
    auxiliary labels are loaded when present.

    Parameters
    ----------
    annotation_file:
        Path to the Hateful Memes JSONL annotation file.

    image_dir:
        Directory containing the dataset images.

    Returns
    -------
    List[MultimodalSample]
        Samples converted to the project's canonical schema.
    """

    annotation_path = Path(annotation_file)
    image_root = Path(image_dir)

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Hateful Memes annotation file not found: {annotation_path}"
        )

    if not image_root.exists():
        raise FileNotFoundError(
            f"Hateful Memes image directory not found: {image_root}"
        )

    samples = []

    with annotation_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {annotation_path}"
                ) from exc

            # ----------------------------------------------------
            # ORIGINAL HATEFUL MEMES FIELDS
            # ----------------------------------------------------

            sample_id = str(record["id"])

            text = record.get("text")

            hate_label = record.get("label")

            image_relative_path = record.get("img")

            image_path = None

            if image_relative_path:
                image_path = str(
                    image_root / image_relative_path
                )

            # ----------------------------------------------------
            # AUXILIARY LABELS
            # ----------------------------------------------------

            sarcasm_label = record.get(
                "sarcasm_label"
            )

            target_group = record.get(
                "target_group"
            )

            # ----------------------------------------------------
            # METADATA
            # ----------------------------------------------------

            metadata = {
                key: value
                for key, value in record.items()
                if key not in {
                    "id",
                    "img",
                    "text",
                    "label",
                    "sarcasm_label",
                    "target_group",
                }
            }

            # ----------------------------------------------------
            # CANONICAL SAMPLE
            # ----------------------------------------------------

            samples.append(
                MultimodalSample(
                    sample_id=sample_id,
                    source="hateful_memes",
                    image_path=image_path,
                    text=text,
                    hate_label=hate_label,
                    sarcasm_label=sarcasm_label,
                    target_group=target_group,
                    metadata=metadata,
                )
            )

    return samples