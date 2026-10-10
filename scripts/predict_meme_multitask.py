
from pathlib import Path
import argparse
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodal_hate.models.multimodal_model import MultimodalHateSpeechModel
from multimodal_hate.models.sarcasm.sarcasm_bert import SarcasmBERT
from multimodal_hate.models.sarcasm.sentiment import SentimentReversal
from multimodal_hate.inference.predictor import MultimodalPredictor
from multimodal_hate.data.processing.ocr import extract_text_from_image


CHECKPOINT_PATH = (
    PROJECT_ROOT / "artifacts"
    / "checkpoints_multitask_sarcasm_priority"
    / "best.pt"
)

SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
TARGET_GROUPS = ("Race", "Religion", "Gender", "Sexuality")


def load_model(device):
    sarcasm_encoder = SarcasmBERT(
        model_name="bert-base-uncased",
        representation_dim=768,
        max_length=128,
        dropout=0.1,
    )

    sentiment_tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL_NAME
    )
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_NAME
    )

    sentiment_module = SentimentReversal(
        tokenizer=sentiment_tokenizer,
        model=sentiment_model,
        device=device,
    )

    model = MultimodalHateSpeechModel(
        sarcasm_encoder=sarcasm_encoder,
        sentiment_module=sentiment_module,
        fusion_dim=512,
        sarcasm_gate_dim=64,
        num_fusion_layers=2,
        num_heads=8,
        dropout=0.1,
    )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("Loading multitask checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    return model.to(device).eval()


def prepare_text(image_path, supplied_text, device, review_ocr):
    if supplied_text is not None:
        text = supplied_text
        print("Using supplied meme text.")
    else:
        print("Extracting text from image using OCR...")
        text = extract_text_from_image(
            str(image_path),
            gpu=(device.type == "cuda"),
        )

    text = " ".join(text.split())

    print("\nOCR / input text:")
    print(text if text else "[No text detected]")

    if review_ocr and supplied_text is None:
        print(
            "\nIf the OCR is incorrect, enter the corrected text."
            "\nPress Enter to keep the OCR result."
        )
        try:
            corrected = input("Corrected text: ").strip()
        except (EOFError, KeyboardInterrupt):
            corrected = ""

        if corrected:
            text = corrected
            print("Using corrected text.")

    if not text:
        print(
            "\nWARNING: No text was detected. "
            "The prediction may be unreliable."
        )

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Sarcasm, hate speech, and target-group analysis"
    )
    parser.add_argument("image", help="Path to the meme image")
    parser.add_argument(
        "--text",
        default=None,
        help="Supply corrected meme text instead of using OCR",
    )
    parser.add_argument(
        "--review-ocr",
        action="store_true",
        help="Review and optionally correct OCR text before prediction",
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    text = prepare_text(
        image_path=image_path,
        supplied_text=args.text,
        device=device,
        review_ocr=args.review_ocr,
    )

    model = load_model(device)
    predictor = MultimodalPredictor(model=model, device=device)

    with torch.inference_mode():
        result = predictor.predict_from_path(
            image_path=str(image_path),
            text=text,
            sample_id=image_path.name,
        )

    print("\n========== MULTITASK PREDICTION ==========")

    print("\n1. SARCASM DETECTION")
    print(
        "Prediction:",
        "Sarcastic" if result.sarcasm_label == 1 else "Not sarcastic",
    )
    print(f"Sarcasm probability: {result.sarcasm_probability:.4f}")

    print("\n2. HATE SPEECH DETECTION")
    print(
        "Prediction:",
        "Hate speech" if result.hate_label == 1 else "Not hate speech",
    )
    print(f"Hate probability: {result.hate_probability:.4f}")

    print("\n3. TARGET-GROUP ANALYSIS")
    found_target = False

    for group, probability, label in zip(
        TARGET_GROUPS,
        result.target_probabilities,
        result.target_labels,
    ):
        print(
            f"{group}: {'Detected' if label else 'Not detected'} "
            f"(probability: {probability:.4f})"
        )
        found_target = found_target or bool(label)

    if not found_target:
        print("No target group crossed the current 0.5 threshold.")

    print(
        "\nNote: Predictions are model outputs, not definitive judgments. "
        "Review OCR accuracy and meme context when evaluating results."
    )


if __name__ == "__main__":
    main()
