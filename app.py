
import sys
import uuid
import gc
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO

import torch
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

# Project paths
ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_targets import create_model
from src.multimodal_hate.inference.predictor import MultimodalPredictor
from src.multimodal_hate.data.processing.ocr import extract_text_from_image


# Use exactly the checkpoints from scripts/predict_meme.py
BASELINE_CHECKPOINT = (
    ROOT / "artifacts" / "checkpoints_fusion2" / "best.pt"
)

SARCASM_CHECKPOINT = (
    ROOT / "artifacts" / "checkpoints_mmsd_sarcasm" / "best.pt"
)

SARCASM_THRESHOLD = 0.25
TARGET_GROUPS = ("race", "religion", "gender", "sexuality")
TARGET_NAMES = ("Race", "Religion", "Gender", "Sexuality")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Sarcasm-Aware Multimodal Hate Speech Detection")

# Avoid simultaneous model loading and inference on the same GPU.
prediction_lock = threading.Lock()


def get_value(obj, *names, default=None):
    """Read a value from either a dictionary or a prediction object."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def load_model(checkpoint_path):
    """Load a model using the same function and weights as the CLI script."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = create_model(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def clear_model(model=None, predictor=None):
    """Release model memory between the two prediction stages."""
    if predictor is not None:
        del predictor

    if model is not None:
        del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_target_probabilities(prediction):
    """Convert predictor target scores into the frontend's expected format."""
    raw_targets = get_value(
        prediction,
        "target_probabilities",
        "target_group_probabilities",
        "target_probs",
        default={},
    )

    if isinstance(raw_targets, (list, tuple)):
        raw_targets = {
            name: raw_targets[index]
            for index, name in enumerate(TARGET_NAMES)
            if index < len(raw_targets)
        }

    if not isinstance(raw_targets, dict):
        raw_targets = {}

    probabilities = {}

    for lower_name, upper_name in zip(TARGET_GROUPS, TARGET_NAMES):
        value = raw_targets.get(
            upper_name,
            raw_targets.get(lower_name),
        )

        if value is not None:
            probabilities[lower_name] = float(value)

    labels = {
        group: int(probabilities.get(group, 0.0) >= 0.5)
        for group in TARGET_GROUPS
    }

    return probabilities, labels


def run_prediction(image_path, ocr_text):
    """
    Follow the same two-stage inference pipeline as predict_meme.py:
    baseline checkpoint for hate/targets, then sarcasm checkpoint.
    """
    with prediction_lock:
        baseline_model = None
        baseline_predictor = None
        sarcasm_model = None
        sarcasm_predictor = None

        try:
            # Stage 1: hate detection and target groups
            baseline_model = load_model(BASELINE_CHECKPOINT)

            baseline_predictor = MultimodalPredictor(
                model=baseline_model,
                device=DEVICE,
            )

            with torch.inference_mode():
                baseline_result = baseline_predictor.predict_from_path(
                    str(image_path),
                    ocr_text,
                    sample_id=Path(image_path).stem,
                )

            hate_probability = float(
                get_value(
                    baseline_result,
                    "hate_probability",
                    "hate_prob",
                    default=0.0,
                )
            )

            hate_label = get_value(
                baseline_result,
                "hate_label",
                "label",
            )

            if not isinstance(hate_label, str):
                hate_label = (
                    "Hate"
                    if hate_probability >= 0.5
                    else "Not hate"
                )

            target_probabilities, target_labels = (
                normalize_target_probabilities(baseline_result)
            )

            # Release the baseline model before loading the next one.
            del baseline_predictor
            baseline_predictor = None

            del baseline_model
            baseline_model = None

            del baseline_result
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 2: dedicated sarcasm model
            sarcasm_model = load_model(SARCASM_CHECKPOINT)

            sarcasm_predictor = MultimodalPredictor(
                model=sarcasm_model,
                device=DEVICE,
            )

            with torch.inference_mode():
                sarcasm_result = sarcasm_predictor.predict_from_path(
                    str(image_path),
                    ocr_text,
                    sample_id=Path(image_path).stem,
                )

            sarcasm_probability = float(
                get_value(
                    sarcasm_result,
                    "sarcasm_probability",
                    "sarcasm_prob",
                    default=0.0,
                )
            )

            sarcasm_label = (
                "Sarcastic"
                if sarcasm_probability >= SARCASM_THRESHOLD
                else "Not sarcastic"
            )

            return {
                "hate": {
                    "label": hate_label,
                    "probability": round(hate_probability, 6),
                },
                "sarcasm": {
                    "label": sarcasm_label,
                    "probability": round(sarcasm_probability, 6),
                    "threshold": SARCASM_THRESHOLD,
                },
                "target_groups": {
                    "probabilities": target_probabilities,
                    "predicted_labels": target_labels,
                },
            }

        finally:
            clear_model(
                model=sarcasm_model,
                predictor=sarcasm_predictor,
            )
            clear_model(
                model=baseline_model,
                predictor=baseline_predictor,
            )


@app.on_event("startup")
def startup():
    """Validate the required checkpoint files before serving requests."""
    missing = [
        path
        for path in (BASELINE_CHECKPOINT, SARCASM_CHECKPOINT)
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Required checkpoint file(s) not found: "
            + ", ".join(str(path) for path in missing)
        )

    print(f"Prediction pipeline configured on {DEVICE}")
    print(f"Baseline checkpoint: {BASELINE_CHECKPOINT}")
    print(f"Sarcasm checkpoint: {SARCASM_CHECKPOINT}")


@app.get("/")
def home():
    page = ROOT / "frontend" / "index.html"

    if not page.is_file():
        raise HTTPException(
            status_code=404,
            detail="Create frontend/index.html first.",
        )

    return FileResponse(page)


@app.get("/api/health")
def health():
    checkpoints_ready = (
        BASELINE_CHECKPOINT.is_file()
        and SARCASM_CHECKPOINT.is_file()
    )

    return {
        "status": "ready" if checkpoints_ready else "missing_checkpoints",
        "device": str(DEVICE),
        "pipeline": "baseline_plus_dedicated_sarcasm",
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload PNG, JPG, JPEG, or WEBP.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Maximum image size is 10 MB.",
        )

    temporary_path = None

    try:
        # Validate the uploaded image.
        with Image.open(BytesIO(contents)) as image:
            image.verify()

        with NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_path = Path(temporary_file.name)

        # Use the same automatic OCR function as predict_meme.py.
        extracted_text = extract_text_from_image(
            str(temporary_path)
        )

        if isinstance(extracted_text, (list, tuple)):
            extracted_text = " ".join(map(str, extracted_text))

        ocr_text = str(extracted_text or "")

        # Run both existing checkpoint pipelines.
        predictions = run_prediction(
            temporary_path,
            ocr_text,
        )

        return {
            "request_id": str(uuid.uuid4()),
            "filename": file.filename,
            "ocr_text": ocr_text,
            "hate": predictions["hate"],
            "sarcasm": predictions["sarcasm"],
            "target_groups": predictions["target_groups"],
            "device": str(DEVICE),
            "note": (
                "Predictions are estimates and may be affected "
                "by OCR errors."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"Prediction error: {type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed ({type(exc).__name__}). "
                "Check the server terminal."
            ),
        ) from exc

    finally:
        await file.close()

        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
