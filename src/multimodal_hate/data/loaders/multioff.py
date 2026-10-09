import json
from pathlib import Path
from typing import List

from src.multimodal_hate.data.schema import MultimodalSample


def load_multioff(
    annotation_file: str,
    image_dir: str,
) -> List[MultimodalSample]:
    """
    Load the MultiOFF dataset from a JSONL annotation file.

    Expected annotation format per line:
    {
        "id": "123",
        "img": "img/123.jpg",
        "text": "example text",
        "label": 1
    }
    """

    annotation_path = Path(annotation_file)
    image_root = Path(image_dir)

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"MultiOFF annotation file not found: {annotation_path}"
        )

    if not image_root.exists():
        raise FileNotFoundError(
            f"MultiOFF image directory not found: {image_root}"
        )

    samples = []

    with annotation_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            sample_id = str(record["id"])
            image_path = image_root / record["img"]

            sample = MultimodalSample(
                sample_id=sample_id,
                source="multioff",
                image_path=str(image_path),
                text=record.get("text"),
                hate_label=record.get("label"),
                metadata={
                    key: value
                    for key, value in record.items()
                    if key not in {"id", "img", "text", "label"}
                },
            )

            samples.append(sample)

    return samples