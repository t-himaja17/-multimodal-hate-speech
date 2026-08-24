"""
Training entry point for the multimodal hate-speech model.

Member 4 ownership.

IMPORTANT:
    The default mode is SAFE.
    Full training must be explicitly requested.

Examples:

    Safe one-batch test:
        python scripts/train.py --smoke-test

    One full epoch:
        python scripts/train.py --epochs 1

    Full training:
        python scripts/train.py --epochs 5
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
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
from src.multimodal_hate.training.trainer import (
    MultimodalTrainer,
)
from scripts.prepare_data import (
    create_dataloaders,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multimodal hate-speech model."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run exactly one training batch.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints",
        help="Directory for model checkpoints.",
    )

    return parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

def resolve_device(
    device_name: str,
) -> torch.device:
    """Resolve requested training device."""

    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# SARCASM MODULE
# ============================================================

def create_sarcasm_encoder(
    device: torch.device,
) -> SarcasmBERT:
    """
    Create the sarcasm encoder.

    The checkpoint remains configurable because the project
    methodology does not specify a single mandatory
    SarcasmBERT checkpoint.
    """

    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = SarcasmBERT(
        model_name=model_name,
        device=device,
        max_length=128,
        dropout=0.1,
        tokenizer=tokenizer,
        representation_dim=768,
    )

    model.to(device)

    return model


# ============================================================
# SENTIMENT MODULE
# ============================================================

def create_sentiment_module(
    device: torch.device,
) -> SentimentReversal:
    """
    Create the sentiment-reversal module.

    A sequence-classification model is required because
    SentimentReversal extracts classification logits.
    """

    model_name = (
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
    )

    model.to(device)

    model.eval()

    return SentimentReversal(
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=128,
        reversal_threshold=0.5,
        vader_threshold=0.05,
    )


# ============================================================
# MODEL
# ============================================================

def create_model(
    device: torch.device,
) -> MultimodalHateSpeechModel:
    """Create the project multimodal model."""

    sarcasm_encoder = create_sarcasm_encoder(
        device
    )

    sentiment_module = create_sentiment_module(
        device
    )

    return MultimodalHateSpeechModel(
        sarcasm_encoder=sarcasm_encoder,
        sentiment_module=sentiment_module,
        fusion_dim=512,
        sarcasm_gate_dim=64,
        num_fusion_layers=4,
        num_heads=8,
        dropout=0.1,
    )


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer."""

    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )


# ============================================================
# SAFE SMOKE TEST
# ============================================================

def run_smoke_test(
    trainer: MultimodalTrainer,
    train_loader,
) -> None:
    """
    Run exactly ONE training batch.

    This deliberately avoids a full epoch.
    """

    print()
    print("=" * 60)
    print("SAFE TRAINING SMOKE TEST")
    print("=" * 60)

    batch = next(iter(train_loader))

    print()
    print("ONE BATCH LOADED")
    print("Images:", batch["image"].shape)
    print("Texts:", len(batch["text"]))

    class OneBatchLoader:
        def __iter__(self):
            yield batch

        def __len__(self):
            return 1

    one_batch_loader = OneBatchLoader()

    result = trainer.train_epoch(
        one_batch_loader
    )

    print()
    print("TRAINING SUCCESS")
    print("Batches:", result.batches)
    print("Total loss:", result.loss)
    print("Hate loss:", result.hate_loss)
    print("Sarcasm loss:", result.sarcasm_loss)
    print("Contrastive loss:", result.contrastive_loss)
    print("Target loss:", result.target_loss)

    print()
    print("=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    set_seed(args.seed)

    device = resolve_device(
        args.device
    )

    print("=" * 60)
    print("MULTIMODAL HATE-SPEECH TRAINING")
    print("=" * 60)

    print()
    print("Device:", device)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)

    if args.smoke_test:
        print("MODE: SAFE SMOKE TEST")
        print("Only ONE batch will be trained.")
    else:
        print("MODE: FULL TRAINING")
        print(
            "WARNING: Full dataset training requested."
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_loader, dev_loader, test_loader = (
        create_dataloaders(
            batch_size=args.batch_size,
            num_workers=0,
        )
    )

    print()
    print(
        "Train samples:",
        len(train_loader.dataset),
    )

    print(
        "Dev samples:",
        len(dev_loader.dataset),
    )

    print(
        "Test samples:",
        len(test_loader.dataset),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("Creating model...")

    model = create_model(
        device
    )

    model.to(device)

    print("MODEL: OK")

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = create_optimizer(
        model
    )

    print("OPTIMIZER: OK")

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        gradient_accumulation_steps=1,
        mixed_precision=(
            device.type == "cuda"
        ),
        gradient_clipping=1.0,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("TRAINER: OK")

    # --------------------------------------------------------
    # SAFE MODE
    # --------------------------------------------------------

    if args.smoke_test:

        run_smoke_test(
            trainer,
            train_loader,
        )

        return

    # --------------------------------------------------------
    # FULL TRAINING
    # --------------------------------------------------------

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=dev_loader,
        epochs=args.epochs,
        early_stopping_patience=5,
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Epochs completed:",
        len(history.train_loss),
    )


if __name__ == "__main__":
    main()