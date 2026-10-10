
from pathlib import Path
import json
from collections import Counter

from datasets import load_dataset

ROOT = Path("data/processed/mmsd_sarcasm")
DATASET = "coderchen01/MMSD2.0"
CONFIG = "mmsd-v2"

# MMSD2.0 label convention: 1 = sarcastic, 0 = not sarcastic.
# The script prints the label counts so they can be checked.
splits = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}

for output_split, source_split in splits.items():
    print(f"\nLoading {source_split}...")
    ds = load_dataset(DATASET, name=CONFIG, split=source_split)

    image_dir = ROOT / output_split / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = ROOT / f"{output_split}.jsonl"
    counts = Counter()
    written = 0

    with output_jsonl.open("w", encoding="utf-8") as out:
        for i, row in enumerate(ds):
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Unexpected label: {label}")

            image = row["image"]
            if image is None:
                continue

            sample_id = str(row.get("id", f"{output_split}_{i}"))
            safe_id = "".join(
                c if c.isalnum() or c in "-_" else "_"
                for c in sample_id
            )
            image_path = image_dir / f"{i}_{safe_id}.jpg"

            image.convert("RGB").save(image_path, format="JPEG")

            record = {
                "id": sample_id,
                "img": image_path.as_posix(),
                "text": str(row.get("text", "")),
                "sarcasm_label": label,
                "label_source": "MMSD2.0",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[label] += 1
            written += 1

    print(f"Split: {output_split}")
    print(f"Exported: {written}")
    print(f"Label counts: {dict(counts)}")
    print(f"Manifest: {output_jsonl}")

print("\nDataset export complete.")
