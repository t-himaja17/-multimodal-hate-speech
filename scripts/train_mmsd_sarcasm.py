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

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "mmsd_sarcasm"
INITIAL_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints_fusion2" / "best.pt"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints_mmsd_sarcasm"


class MMSDSarcasmDataset(Dataset):
    def __init__(self, split):
        annotation_file = DATA_DIR / f"{split}.jsonl"
        if not annotation_file.is_file():
            raise FileNotFoundError(f"Missing file: {annotation_file}")

        self.records = []
        with annotation_file.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                record["image_path"] = PROJECT_ROOT / record["img"]
                label = int(record["sarcasm_label"])
                if label not in (0, 1):
                    raise ValueError(f"Invalid sarcasm label: {label}")
                self.records.append(record)

        if not self.records:
            raise ValueError(f"No samples found in {annotation_file}")

        print(f"{split}: {len(self.records)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = record["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as source:
            image = source.convert("RGB").resize(
                (224, 224), Image.Resampling.BICUBIC
            )
            image_tensor = torch.from_numpy(
                np.array(image).copy()
            ).permute(2, 0, 1).float() / 255.0

        label = int(record["sarcasm_label"])
        sarcasm_target = torch.zeros(2, dtype=torch.float32)
        sarcasm_target[label] = 1.0

        return {
            "image": image_tensor,
            "text": str(record.get("text") or ""),
            "hate_target": torch.zeros(2, dtype=torch.float32),
            "sarcasm_target": sarcasm_target,
            "target_target": torch.zeros(4, dtype=torch.float32),
            "sample_id": str(record.get("id", index)),
        }


def collate_batch(batch):
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "text": [x["text"] for x in batch],
        "hate_target": torch.stack([x["hate_target"] for x in batch]),
        "sarcasm_target": torch.stack([x["sarcasm_target"] for x in batch]),
        "target_target": torch.stack([x["target_target"] for x in batch]),
        "sample_id": [x["sample_id"] for x in batch],
    }


def create_model(device):
    sarcasm_encoder = SarcasmBERT(
        model_name="bert-base-uncased",
        representation_dim=768,
        max_length=128,
        dropout=0.1,
    )

    sentiment_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_name)
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

    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {INITIAL_CHECKPOINT}")

    checkpoint = torch.load(
        INITIAL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
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
    pretrained_params, task_params = [], []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(key in name for key in pretrained_keywords):
            pretrained_params.append(parameter)
        else:
            task_params.append(parameter)

    groups = []
    if pretrained_params:
        groups.append({"params": pretrained_params, "lr": 1e-5})
    if task_params:
        groups.append({"params": task_params, "lr": 1e-4})

    if not groups:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(groups, weight_decay=0.01)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError("Use epochs >= 1 and batch-size >= 2.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = MMSDSarcasmDataset("train")
    validation_dataset = MMSDSarcasmDataset("validation")

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
        hate_weight=0.0,
        sarcasm_weight=1.0,
        contrastive_weight=0.0,
        target_weight=0.0,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
        mixed_precision=(device.type == "cuda"),
        gradient_clipping=1.0,
    )

    print("Starting MMSD2.0 sarcasm training...")
    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=args.epochs,
    )

    print("Training finished.")
    print("Epochs completed:", len(history.train))
    print("New checkpoints:", CHECKPOINT_DIR)
    print("Original checkpoint preserved:", INITIAL_CHECKPOINT)


if __name__ == "__main__":
    main()