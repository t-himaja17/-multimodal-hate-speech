import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import torch

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.multimodal_hate.data.loaders.hateful_memes import (
    load_hateful_memes,
)
from src.multimodal_hate.training.dataset import (
    MultimodalHateSpeechDataset,
    multimodal_collate_fn,
)
from src.multimodal_hate.training.trainer import (
    MultimodalTrainer,
)
from src.multimodal_hate.training.losses import (
    MultimodalTotalLoss,
)
from src.multimodal_hate.models.multimodal_model import (
    MultimodalHateSpeechModel,
)
from src.multimodal_hate.models.sarcasm.sarcasm_bert import (
    SarcasmBERT,
)
from src.multimodal_hate.models.sarcasm.sentiment import (
    SentimentReversal,
)


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "hateful_memes"

TRAIN_FILE = DATA_ROOT / "train_weak_labels.jsonl"
DEV_FILE = DATA_ROOT / "dev_weak_labels.jsonl"
TEST_FILE = DATA_ROOT / "test.jsonl"

IMAGE_DIR = DATA_ROOT


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multimodal hate speech model"
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints_sarcasm_weighted",
    )

    parser.add_argument(
        "--sarcasm-positive-weight",
        type=float,
        default=10.0,
        help="Weight applied to positive sarcasm targets.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# LOAD DATA
# ============================================================

def create_dataloaders(batch_size):
    for path in (TRAIN_FILE, DEV_FILE, TEST_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required annotation file not found: {path}"
            )

    train_samples = load_hateful_memes(
        str(TRAIN_FILE), str(IMAGE_DIR)
    )
    dev_samples = load_hateful_memes(
        str(DEV_FILE), str(IMAGE_DIR)
    )
    test_samples = load_hateful_memes(
        str(TEST_FILE), str(IMAGE_DIR)
    )

    print("Train samples:", len(train_samples))
    print("Dev samples:", len(dev_samples))
    print("Test samples:", len(test_samples))

    train_dataset = MultimodalHateSpeechDataset(train_samples)
    dev_dataset = MultimodalHateSpeechDataset(dev_samples)
    test_dataset = MultimodalHateSpeechDataset(test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )

    return train_loader, dev_loader, test_loader


# ============================================================
# FREEZE HATEBERT
# ============================================================

def freeze_hatebert(model):
    if not hasattr(model, "text_encoder"):
        print("WARNING: model.text_encoder not found.")
        return

    count = 0

    for parameter in model.text_encoder.parameters():
        parameter.requires_grad = False
        count += parameter.numel()

    model.text_encoder.eval()

    print("HateBERT frozen parameters:", count)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(device):
    print("\n" + "=" * 70)
    print("CREATING 2-LAYER FUSION MODEL")
    print("=" * 70)

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

    model.to(device)

    freeze_hatebert(model)

    if hasattr(model, "sentiment_module"):
        sentiment_model_obj = model.sentiment_module.model

        for parameter in sentiment_model_obj.parameters():
            parameter.requires_grad = False

        sentiment_model_obj.eval()
        print("Sentiment model frozen.")

    return model


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(model):
    pretrained_parameters = []
    task_parameters = []
    frozen_parameters = []

    pretrained_keywords = (
        "clip_image_encoder",
        "clip_text_encoder",
        "image_encoder",
        "sarcasm_encoder",
    )

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen_parameters.append(parameter)
            continue

        if any(key in name for key in pretrained_keywords):
            pretrained_parameters.append(parameter)
        else:
            task_parameters.append(parameter)

    groups = []

    if pretrained_parameters:
        groups.append({
            "params": pretrained_parameters,
            "lr": 1e-5,
        })

    if task_parameters:
        groups.append({
            "params": task_parameters,
            "lr": 1e-4,
        })

    if not groups:
        raise RuntimeError("No trainable model parameters found.")

    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=0.01,
    )

    print("\nOPTIMIZER PARAMETER GROUPS")
    print(
        "Pretrained encoder parameters:",
        sum(p.numel() for p in pretrained_parameters),
    )
    print(
        "Task-specific parameters:",
        sum(p.numel() for p in task_parameters),
    )
    print(
        "Frozen parameters:",
        sum(p.numel() for p in frozen_parameters),
    )

    return optimizer


# ============================================================
# LOSS WITH SARCASM CLASS WEIGHTING
# ============================================================

def create_loss(sarcasm_positive_weight):
    print("\nLOSS CONFIGURATION")
    print("Hate weight: 1.0")
    print("Sarcasm loss weight: 0.05")
    print("Sarcasm positive-class weight:", sarcasm_positive_weight)
    print("Contrastive weight: 0.01")
    print("Target weight: 0.05")

    return MultimodalTotalLoss(
        hate_weight=1.0,
        sarcasm_weight=0.05,
        contrastive_weight=0.01,
        target_weight=0.05,
        sarcasm_positive_weight=sarcasm_positive_weight,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2.")

    if args.sarcasm_positive_weight <= 0:
        raise ValueError(
            "--sarcasm-positive-weight must be greater than zero."
        )

    device = torch.device(
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_dir = Path(args.checkpoint_dir)

    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir

    # Prevent accidental overwriting of the original model.
    original_checkpoint_dir = (
        PROJECT_ROOT / "artifacts" / "checkpoints_fusion2"
    ).resolve()

    if checkpoint_dir.resolve() == original_checkpoint_dir:
        raise ValueError(
            "Choose a new checkpoint directory. "
            "The original fusion2 checkpoint must be preserved."
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SARCASM-WEIGHTED MULTIMODAL TRAINING")
    print("=" * 70)
    print("Device:", device)
    print("Fusion layers: 2")
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Checkpoint directory:", checkpoint_dir)

    train_loader, dev_loader, _ = create_dataloaders(
        args.batch_size
    )

    model = create_model(device)
    optimizer = create_optimizer(model)

    loss_fn = create_loss(
        args.sarcasm_positive_weight
    )

    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        checkpoint_dir=checkpoint_dir,
        mixed_precision=(device.type == "cuda"),
        gradient_clipping=1.0,
    )

    if args.smoke_test:
        print("\nRunning one-epoch smoke test...")
        trainer.fit(
            train_loader=train_loader,
            validation_loader=train_loader,
            epochs=1,
        )
        print("Smoke test complete.")
        return

    print("\nStarting sarcasm-weighted training...")

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=dev_loader,
        epochs=args.epochs,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("Epochs completed:", len(history.train))
    print("Checkpoint directory:", checkpoint_dir)
    print(
        "The official Hateful Memes test split has no labels; "
        "evaluate hate detection on the labeled dev split."
    )


if __name__ == "__main__":
    main()