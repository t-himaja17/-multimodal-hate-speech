
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from scripts.evaluate_targets import create_model
from src.multimodal_hate.inference.predictor import MultimodalPredictor
from src.multimodal_hate.data.processing.ocr import (
    extract_text_from_image,
    format_ocr_text,
)

DATA = ROOT / "data" / "raw" / "hateful_memes"
ANNOTATIONS = DATA / "dev_weak_labels.jsonl"
CHECKPOINT = ROOT / "artifacts" / "checkpoints_fusion2" / "best.pt"
GROUPS = ["race", "religion", "gender", "sexuality"]


def read_rows():
    with ANNOTATIONS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_samples(rows):
    selected = []
    used = set()

    def choose(description, predicate):
        candidates = [
            r for r in rows
            if str(r["id"]) not in used
            and predicate(r)
            and (DATA / r["img"]).is_file()
        ]
        if not candidates:
            print(f"No available sample found for: {description}")
            return

        # Prefer records with explicit sarcasm annotations when possible.
        candidates.sort(
            key=lambda r: (
                r.get("sarcasm_label") is None,
                str(r["id"]),
            )
        )
        row = candidates[0]
        used.add(str(row["id"]))
        selected.append((description, row))

    for group in GROUPS:
        choose(
            f"Target: {group}",
            lambda r, g=group: g in (r.get("target_group") or []),
        )

    choose("Official hate example", lambda r: r.get("label") == 1)
    choose("Official non-hate example", lambda r: r.get("label") == 0)
    choose(
        "Annotated sarcastic example",
        lambda r: r.get("sarcasm_label") == 1,
    )
    choose(
        "Annotated non-sarcastic example",
        lambda r: r.get("sarcasm_label") == 0,
    )

    return selected


def main():
    if not ANNOTATIONS.exists():
        raise FileNotFoundError(ANNOTATIONS)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    rows = read_rows()
    samples = select_samples(rows)

    if not samples:
        raise RuntimeError("No usable test images found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictor = MultimodalPredictor(model=model, device=device)
    results = []

    for description, row in samples:
        image_path = DATA / row["img"]

        print("\n" + "=" * 65)
        print(f"TEST: {description} | ID: {row['id']}")
        print("Text in annotation:", row.get("text", ""))
        print("Official hate label:", row.get("label"))
        print("Weak target labels:", row.get("target_group"))
        print("Sarcasm annotation:", row.get("sarcasm_label"))

        ocr_result = extract_text_from_image(str(image_path))
        if isinstance(ocr_result, str):
            ocr_text = ocr_result.strip()
        else:
            ocr_text = format_ocr_text(ocr_result).strip()

        print("OCR:", ocr_text or "[No text detected]")

        prediction = predictor.predict_from_path(
            str(image_path),
            ocr_text,
            sample_id=str(row["id"]),
        )

        hate_p = float(prediction.hate_probability)
        sarcasm_p = float(prediction.sarcasm_probability)
        target_p = [float(x) for x in prediction.target_probabilities]
        target_labels = list(prediction.target_labels)

        print(
            f"Predicted hate: "
            f"{'HATE' if prediction.hate_label == 1 else 'NOT HATE'} "
            f"(hate probability {hate_p:.1%})"
        )
        print(
            f"Predicted sarcasm: "
            f"{'SARCASTIC' if prediction.sarcasm_label == 1 else 'NOT SARCASTIC'} "
            f"(sarcasm probability {sarcasm_p:.1%})"
        )
        print("Target-group probabilities:")
        for group, probability, label in zip(
            GROUPS, target_p, target_labels
        ):
            print(
                f"  {group:<10} {probability:.1%} "
                f"{'<-- SELECTED' if label else ''}"
            )

        results.append({
            "id": row["id"],
            "test_category": description,
            "official_hate_label": row.get("label"),
            "weak_target_labels": row.get("target_group"),
            "sarcasm_annotation": row.get("sarcasm_label"),
            "ocr_text": ocr_text,
            "predicted_hate_label": int(prediction.hate_label),
            "hate_probability": hate_p,
            "predicted_sarcasm_label": int(prediction.sarcasm_label),
            "sarcasm_probability": sarcasm_p,
            "target_probabilities": dict(zip(GROUPS, target_p)),
            "predicted_target_groups": [
                group for group, label in zip(GROUPS, target_labels) if label
            ],
        })

    output = ROOT / "artifacts" / "varied_meme_test_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    print(f"Tested {len(results)} images.")
    print("Full results saved to:", output)


if __name__ == "__main__":
    main()
