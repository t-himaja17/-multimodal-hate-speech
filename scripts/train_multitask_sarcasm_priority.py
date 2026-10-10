import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.multimodal_hate.models.multimodal_model import MultimodalHateSpeechModel
from src.multimodal_hate.models.sarcasm.sarcasm_bert import SarcasmBERT
from src.multimodal_hate.models.sarcasm.sentiment import SentimentReversal
from src.multimodal_hate.training.trainer import MultimodalTrainer
from src.multimodal_hate.training.losses import MultimodalTotalLoss

MMSD_DIR = PROJECT_ROOT / "data" / "processed" / "mmsd_sarcasm"
HATEFUL_MEMES_DIR = PROJECT_ROOT / "data" / "raw" / "hateful_memes"

BASE_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "checkpoints_mmsd_sarcasm" / "best.pt"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT / "artifacts" / "checkpoints_multitask_sarcasm_priority"
)

TARGET_GROUPS = ("race", "religion", "gender", "sexuality")


def one_hot_binary(label):
    target = torch.zeros(2, dtype=torch.float32)

    if label is not None:
        label = int(label)

        if label not in (0, 1):
            raise ValueError(f"Invalid binary label: {label}")

        target[label] = 1.0

    return target


def target_vector(groups):
    target = torch.zeros(4, dtype=torch.float32)

    if not groups:
        return target

    if isinstance(groups, str):
        groups = [groups]

    if not isinstance(groups, (list, tuple, set)):
        return target

    group_to_index = {
        name: idx for idx, name in enumerate(TARGET_GROUPS)
    }

    for group in groups:
        if isinstance(group, str):
            index = group_to_index.get(group.strip().lower())

            if index is not None:
                target[index] = 1.0

    return target


def load_image_tensor(image_path):
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as source:
        image = source.convert("RGB").resize(
            (224, 224),
            Image.Resampling.BICUBIC,
        )

        return (
            torch.from_numpy(np.array(image).copy())
            .permute(2, 0, 1)
            .float()
            / 255.0
        )


class MultiTaskDataset(Dataset):

    def __init__(self, split):
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")

        self.records = []

        annotation_file = MMSD_DIR / f"{split}.jsonl"

        if not annotation_file.is_file():
            raise FileNotFoundError(
                f"Missing MMSD2.0 file: {annotation_file}"
            )

        with annotation_file.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)

                self.records.append({
                    "source": "mmsd",
                    "id": str(row.get("id", len(self.records))),
                    "image_path": PROJECT_ROOT / row["img"],
                    "text": str(row.get("text") or ""),
                    "hate_target": one_hot_binary(None),
                    "sarcasm_target": one_hot_binary(
                        row.get("sarcasm_label")
                    ),
                    "target_target": torch.zeros(
                        4, dtype=torch.float32
                    ),
                })

        # Hateful Memes weak labels are used only for training.
        if split == "train":
            weak_file = (
                HATEFUL_MEMES_DIR / "train_weak_labels.jsonl"
            )

            if not weak_file.is_file():
                raise FileNotFoundError(
                    f"Missing weak-label file: {weak_file}"
                )

            with weak_file.open(encoding="utf-8") as file:
                for line in file:
                    row = json.loads(line)

                    relative_image = Path(row["img"])

                    if relative_image.is_absolute():
                        image_path = relative_image
                    else:
                        image_path = (
                            HATEFUL_MEMES_DIR / relative_image
                        )

                    self.records.append({
                        "source": "hateful_memes_weak",
                        "id": str(row.get("id", len(self.records))),
                        "image_path": image_path,
                        "text": str(row.get("text") or ""),
                        "hate_target": one_hot_binary(
                            row.get("label")
                        ),
                        # Ignore noisy, imbalanced sarcasm pseudo-labels.
                        "sarcasm_target": one_hot_binary(None),
                        "target_target": target_vector(
                            row.get("target_group")
                        ),
                    })

        if not self.records:
            raise ValueError(f"No records found for {split}")

        print(f"{split}: {len(self.records)} samples")

        if split == "train":
            mmsd_count = sum(
                row["source"] == "mmsd"
                for row in self.records
            )
            hateful_count = len(self.records) - mmsd_count

            print(f"  MMSD2.0 samples: {mmsd_count}")
            print(
                f"  Hateful Memes weak-label samples: {hateful_count}"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]

        return {
            "image": load_image_tensor(row["image_path"]),
            "text": row["text"],
            "hate_target": row["hate_target"].clone(),
            "sarcasm_target": row["sarcasm_target"].clone(),
            "target_target": row["target_target"].clone(),
            "sample_id": row["id"],
        }


def collate_batch(batch):
    return {
        "image": torch.stack([
            item["image"] for item in batch
        ]),
        "text": [
            item["text"] for item in batch
        ],
        "hate_target": torch.stack([
            item["hate_target"] for item in batch
        ]),
        "sarcasm_target": torch.stack([
            item["sarcasm_target"] for item in batch
        ]),
        "target_target": torch.stack([
            item["target_target"] for item in batch
        ]),
        "sample_id": [
            item["sample_id"] for item in batch
        ],
    }


def create_model(device):
    sarcasm_encoder = SarcasmBERT(
        model_name="bert-base-uncased",
        representation_dim=768,
        max_length=128,
        dropout=0.1,
    )

    sentiment_name = (
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    sentiment_tokenizer = AutoTokenizer.from_pretrained(
        sentiment_name
    )

    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        sentiment_name
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

    if not BASE_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Best sarcasm checkpoint not found: {BASE_CHECKPOINT}"
        )

    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model.load_state_dict(state_dict, strict=True)

    for parameter in model.text_encoder.parameters():
        parameter.requires_grad = False

    for parameter in model.sentiment_module.model.parameters():
        parameter.requires_grad = False

    return model.to(device)


def create_optimizer(model):
    pretrained_keywords = (
        "clip_image_encoder",
        "clip_text_encoder",
        "image_encoder",
        "sarcasm_encoder",
    )

    pretrained_params = []
    task_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if any(key in name for key in pretrained_keywords):
            pretrained_params.append(parameter)
        else:
            task_params.append(parameter)

    groups = []

    if pretrained_params:
        groups.append({
            "params": pretrained_params,
            "lr": 1e-5,
        })

    if task_params:
        groups.append({
            "params": task_params,
            "lr": 1e-4,
        })

    if not groups:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(
        groups,
        weight_decay=0.01,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-task training with sarcasm as the primary objective."
        )
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError(
            "Use epochs >= 1 and batch-size >= 2."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Starting checkpoint:", BASE_CHECKPOINT)
    print("Output directory:", CHECKPOINT_DIR)

    train_dataset = MultiTaskDataset("train")
    validation_dataset = MultiTaskDataset("validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(device)
    optimizer = create_optimizer(model)

    loss_fn = MultimodalTotalLoss(
        hate_weight=0.35,
        sarcasm_weight=1.0,
        contrastive_weight=0.0,
        target_weight=0.15,
        sarcasm_positive_weight=1.0,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
        mixed_precision=(device.type == "cuda"),
        gradient_clipping=1.0,
    )

    print("Starting multi-task training.")

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=args.epochs,
        early_stopping_patience=2,
    )

    print("Training finished.")
    print("Epochs completed:", len(history.train))
    print("New checkpoints:", CHECKPOINT_DIR)
    print("Original checkpoint preserved:", BASE_CHECKPOINT)


if __name__ == "__main__":
    main()