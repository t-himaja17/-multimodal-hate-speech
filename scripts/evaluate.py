"""
Evaluation entry point for the multimodal hate-speech model.

Member 4 ownership.

This script:
    1. Loads Hateful Memes dev/test data.
    2. Builds the same model architecture used during training.
    3. Loads a trained checkpoint.
    4. Runs evaluation.
    5. Prints the configured classification metrics.

IMPORTANT:
    This script does NOT train the model.

Examples:

    Evaluate best checkpoint on dev:
        python scripts/evaluate.py --checkpoint artifacts/checkpoints/best.pt --split dev

    Evaluate best checkpoint on test:
        python scripts/evaluate.py --checkpoint artifacts/checkpoints/best.pt --split test

    Force CPU:
        python scripts/evaluate.py --checkpoint artifacts/checkpoints/best.pt --split dev --device cpu
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

from scripts.prepare_data import create_dataloaders
from src.multimodal_hate.models.multimodal_model import (
    MultimodalHateSpeechModel,
)
from src.multimodal_hate.models.sarcasm.sarcasm_bert import (
    SarcasmBERT,
)
from src.multimodal_hate.models.sarcasm.sentiment import (
    SentimentReversal,
)
from src.multimodal_hate.evaluation.evaluator import (
    MultimodalEvaluator,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible evaluation."""

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
        description="Evaluate multimodal hate-speech model."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained checkpoint.",
    )

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Evaluation batch size.",
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
        help="Evaluation device.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary classification threshold.",
    )

    return parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

def resolve_device(
    device_name: str,
) -> torch.device:
    """Resolve requested evaluation device."""

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
    """Create the same sarcasm encoder used during training."""

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
    """Create the same sentiment module used during training."""

    model_name = (
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
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
    """Create the same model architecture used during training."""

    sarcasm_encoder = create_sarcasm_encoder(
        device
    )

    sentiment_module = create_sentiment_module(
        device
    )

    model = MultimodalHateSpeechModel(
        sarcasm_encoder=sarcasm_encoder,
        sentiment_module=sentiment_module,
        fusion_dim=512,
        sarcasm_gate_dim=64,
        num_fusion_layers=4,
        num_heads=8,
        dropout=0.1,
    )

    model.to(device)

    return model


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    set_seed(args.seed)

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be greater than zero."
        )

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    device = resolve_device(
        args.device
    )

    print("=" * 60)
    print("MULTIMODAL HATE-SPEECH EVALUATION")
    print("=" * 60)

    print()
    print("Device:", device)
    print("Split:", args.split)
    print("Batch size:", args.batch_size)
    print("Checkpoint:", args.checkpoint)
    print("Threshold:", args.threshold)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print()
    print("Loading data...")

    train_loader, dev_loader, test_loader = (
        create_dataloaders(
            batch_size=args.batch_size,
            num_workers=0,
        )
    )

    if args.split == "dev":
        evaluation_loader = dev_loader
    else:
        evaluation_loader = test_loader

    print(
        "Evaluation samples:",
        len(evaluation_loader.dataset),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("Creating model...")

    model = create_model(
        device
    )

    print("MODEL: OK")

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    # --------------------------------------------------------
    # TRAINER
    #
    # We use the existing Trainer only for its checkpoint
    # loading implementation.
    # --------------------------------------------------------

    from src.multimodal_hate.training.trainer import (
        MultimodalTrainer,
    )

    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        gradient_accumulation_steps=1,
        mixed_precision=False,
        gradient_clipping=1.0,
        checkpoint_dir="artifacts/checkpoints",
    )

    print("TRAINER: OK")

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    print()
    print("Loading checkpoint...")

    checkpoint = trainer.load_checkpoint(
        args.checkpoint
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch"),
    )

    print("CHECKPOINT: OK")

    # --------------------------------------------------------
    # EVALUATOR
    # --------------------------------------------------------

    evaluator = MultimodalEvaluator(
        model=model,
        device=device,
        threshold=args.threshold,
    )

    print("EVALUATOR: OK")

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)

    results = evaluator.evaluate(
        evaluation_loader
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print()
    print("HATE")
    print("-" * 60)

    for name, value in results["metrics"]["hate"].items():
        print(
            f"{name}: {value:.6f}"
            if not np.isnan(value)
            else f"{name}: NaN"
        )

    print()
    print("SARCASM")
    print("-" * 60)

    for name, value in results["metrics"]["sarcasm"].items():
        print(
            f"{name}: {value:.6f}"
            if not np.isnan(value)
            else f"{name}: NaN"
        )

    print()
    print("Predictions:", len(results["predictions"]["hate"]))
    print("Evaluation complete.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
