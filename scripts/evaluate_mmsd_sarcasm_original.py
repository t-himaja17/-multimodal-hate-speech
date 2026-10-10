
import sys
from pathlib import Path
from collections import Counter

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.train_mmsd_sarcasm import (
    MMSDSarcasmDataset,
    collate_batch,
    create_model,
)

CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints_mmsd_sarcasm"
    / "best.pt"
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Checkpoint:", CHECKPOINT)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    dataset = MMSDSarcasmDataset("test")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for batch_num, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            texts = batch["text"]

            output = model(images, texts)
            logits = output.logits["sarcasm"]
            predictions = logits.argmax(dim=1)
            labels = batch["sarcasm_target"].argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

            if batch_num % 50 == 0:
                print(
                    f"Processed {len(y_true)}/{len(dataset)} samples"
                )

    print("\nCheckpoint epoch:", checkpoint.get("epoch"))
    print(
        "Checkpoint validation loss:",
        checkpoint.get("best_validation_loss"),
    )
    print("Test samples:", len(y_true))
    print("True label counts:", Counter(y_true))
    print("Predicted label counts:", Counter(y_pred))

    print("\nConfusion Matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Not Sarcastic", "Sarcastic"],
            digits=4,
            zero_division=0,
        )
    )

    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1:", round(f1, 4))


if __name__ == "__main__":
    main()

