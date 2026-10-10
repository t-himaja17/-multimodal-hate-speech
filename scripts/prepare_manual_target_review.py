
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "hateful_memes"
SOURCE = DATA / "dev_weak_labels.jsonl"
OUTPUT = ROOT / "artifacts" / "manual_target_review.csv"

GROUPS = {"race", "religion", "gender", "sexuality"}

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing dataset: {SOURCE}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with SOURCE.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

annotated = [r for r in rows if r.get("target_group")]

fields = [
    "id",
    "image_path",
    "text",
    "official_hate_label",
    "weak_target_groups",
    "verified_hate_label",
    "verified_target_groups",
    "review_status",
    "review_notes",
]

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for r in annotated:
        weak_groups = r.get("target_group") or []
        if isinstance(weak_groups, str):
            weak_groups = [weak_groups]

        unknown = set(weak_groups) - GROUPS
        if unknown:
            print(f"Warning: ID {r['id']} has unknown groups: {unknown}")

        writer.writerow({
            "id": r["id"],
            "image_path": str((DATA / r["img"]).resolve()),
            "text": r.get("text", ""),
            "official_hate_label": r.get("label", ""),
            "weak_target_groups": "|".join(weak_groups),
            "verified_hate_label": "",
            "verified_target_groups": "",
            "review_status": "pending",
            "review_notes": "",
        })

print(f"Created: {OUTPUT}")
print(f"Samples to review: {len(annotated)}")
print("Target groups: race, religion, gender, sexuality")
print("Open the CSV in Excel to review the samples.")
