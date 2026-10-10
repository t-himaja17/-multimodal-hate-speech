
import argparse
import gc
import json
import sys
from pathlib import Path

import torch

# Ensure the project root and scripts directory are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_targets import create_model
from src.multimodal_hate.inference.predictor import MultimodalPredictor
from src.multimodal_hate.data.processing.ocr import extract_text_from_image


BASELINE_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints_fusion2/best.pt"
SARCASM_CHECKPOINT = PROJECT_ROOT / "artifacts/checkpoints_mmsd_sarcasm/best.pt"

# Updated threshold: 25%
SARCASM_THRESHOLD = 0.25

TARGET_NAMES = ["Race", "Religion", "Gender", "Sexuality"]


def get_value(obj, *names, default=None):
    """Get a value from a dictionary or an object."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def load_model(checkpoint_path, device):
    """Create the project model and load its checkpoint."""
    model = create_model(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Predict hate speech, sarcasm, and target groups for one meme."
    )
    parser.add_argument("image", help="Path to the meme image")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE_CHECKPOINT}")

    if not SARCASM_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Sarcasm checkpoint not found: {SARCASM_CHECKPOINT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("MULTIMODAL MEME ANALYSIS")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Device: {device}")

    # 1. Extract text from the image
    print("\n[1/3] Extracting image text...")

    extracted_text = extract_text_from_image(str(image_path))

    if isinstance(extracted_text, (list, tuple)):
        extracted_text = " ".join(map(str, extracted_text))

    extracted_text = str(extracted_text or "")

    print("OCR TEXT:")
    print(extracted_text)

    # 2. Baseline model: hate and target-group predictions
    print("\n[2/3] Running hate and target-group model...")

    baseline_model = load_model(BASELINE_CHECKPOINT, device)
    baseline_predictor = MultimodalPredictor(
        model=baseline_model,
        device=device,
    )

    with torch.inference_mode():
        baseline_prediction = baseline_predictor.predict_from_path(
            str(image_path),
            extracted_text,
            sample_id=image_path.stem,
        )

    hate_probability = float(
        get_value(
            baseline_prediction,
            "hate_probability",
            "hate_prob",
            default=0.0,
        )
    )

    hate_label = get_value(
        baseline_prediction,
        "hate_label",
        "label",
    )

    if not isinstance(hate_label, str):
        hate_label = "Hate" if hate_probability >= 0.5 else "Not hate"

    raw_targets = get_value(
        baseline_prediction,
        "target_probabilities",
        "target_group_probabilities",
        "target_probs",
        default={},
    )

    if isinstance(raw_targets, (list, tuple)):
        raw_targets = {
            name: raw_targets[i]
            for i, name in enumerate(TARGET_NAMES)
            if i < len(raw_targets)
        }

    if not isinstance(raw_targets, dict):
        raw_targets = {}

    target_groups = {}

    for name in TARGET_NAMES:
        probability = raw_targets.get(name, raw_targets.get(name.lower()))

        if probability is not None:
            probability = float(probability)
            target_groups[name] = {
                "probability": round(probability, 6),
                "selected": probability >= 0.5,
            }

    del baseline_predictor, baseline_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Dedicated sarcasm model
    print("\n[3/3] Running dedicated sarcasm model...")

    sarcasm_model = load_model(SARCASM_CHECKPOINT, device)
    sarcasm_predictor = MultimodalPredictor(
        model=sarcasm_model,
        device=device,
    )

    with torch.inference_mode():
        sarcasm_prediction = sarcasm_predictor.predict_from_path(
            str(image_path),
            extracted_text,
            sample_id=image_path.stem,
        )

    sarcasm_probability = float(
        get_value(
            sarcasm_prediction,
            "sarcasm_probability",
            "sarcasm_prob",
            default=0.0,
        )
    )

    # Apply the updated 25% threshold
    sarcasm_label = (
        "Sarcastic"
        if sarcasm_probability >= SARCASM_THRESHOLD
        else "Not sarcastic"
    )

    result = {
        "image": str(image_path),
        "ocr_text": extracted_text,
        "hate": {
            "label": hate_label,
            "hate_probability": round(hate_probability, 6),
        },
        "sarcasm": {
            "label": sarcasm_label,
            "sarcasm_probability": round(sarcasm_probability, 6),
            "threshold": SARCASM_THRESHOLD,
            "checkpoint": str(SARCASM_CHECKPOINT.relative_to(PROJECT_ROOT)),
        },
        "target_groups": target_groups,
    }

    print("\n" + "=" * 60)
    print("FINAL PREDICTIONS")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    output_path = PROJECT_ROOT / "artifacts" / f"prediction_{image_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved result: {output_path}")


if __name__ == "__main__":
    main()
