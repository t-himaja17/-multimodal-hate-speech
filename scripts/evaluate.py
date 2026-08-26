"""
Evaluation entry point for the multimodal hate-speech model.

Member 4 ownership.

Examples:

    Show help:
        python scripts/evaluate.py --help

    Evaluate best checkpoint:
        python scripts/evaluate.py --checkpoint artifacts/checkpoints/best.pt

    Evaluate latest checkpoint:
        python scripts/evaluate.py --checkpoint artifacts/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from scripts.prepare_data import create_dataloaders

from src.multimodal_hate.evaluation.evaluator import (
    MultimodalEvaluator,
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
        default="artifacts/checkpoints/best.pt",
        help="Path to model checkpoint.",
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

    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="test",
        help="Dataset split to evaluate.",
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
    """Create the sarcasm encoder used during training."""

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
    """Create the sentiment-reversal module used during training."""

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
# OPTIMIZER
# ============================================================

def create_optimizer(
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    """Create optimizer required by checkpoint loader."""

    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    trainer: MultimodalTrainer,
    checkpoint_path: str,
) -> dict:
    """Load a saved training checkpoint."""

    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    print()
    print("Loading checkpoint:")
    print(path)

    checkpoint = trainer.load_checkpoint(
        path
    )

    print("CHECKPOINT: OK")

    return checkpoint


# ============================================================
# DATA
# ============================================================

def select_dataloader(
    train_loader,
    dev_loader,
    test_loader,
    split: str,
):
    """Select the requested dataset split."""

    if split == "train":
        return train_loader

    if split == "dev":
        return dev_loader

    return test_loader


# ============================================================
# REPORT
# ============================================================

def print_metrics(
    results: dict,
) -> None:
    """Print evaluation metrics."""

    metrics = results["metrics"]

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for task_name, task_metrics in metrics.items():

        print()
        print(task_name.upper())

        for metric_name, value in task_metrics.items():

            if isinstance(value, float):

                if np.isnan(value):
                    print(
                        f"  {metric_name}: NaN"
                    )
                else:
                    print(
                        f"  {metric_name}: "
                        f"{value:.4f}"
                    )

            else:
                print(
                    f"  {metric_name}: {value}"
                )

    print()
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be greater than zero."
        )

    set_seed(args.seed)

    device = resolve_device(
        args.device
    )

    print("=" * 60)
    print("MULTIMODAL HATE-SPEECH EVALUATION")
    print("=" * 60)

    print()
    print("Device:", device)
    print("Batch size:", args.batch_size)
    print("Split:", args.split)
    print("Threshold:", args.threshold)
    print("Checkpoint:", args.checkpoint)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print()
    print("Loading evaluation data...")

    train_loader, dev_loader, test_loader = (
        create_dataloaders(
            batch_size=args.batch_size,
            num_workers=0,
        )
    )

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

    evaluation_loader = select_dataloader(
        train_loader,
        dev_loader,
        test_loader,
        args.split,
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

    optimizer = create_optimizer(
        model
    )

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
        checkpoint_dir="artifacts/checkpoints",
    )

    print("TRAINER: OK")

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    load_checkpoint(
        trainer,
        args.checkpoint,
    )

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
    # EVALUATE
    # --------------------------------------------------------

    print()
    print("Running evaluation...")

    results = evaluator.evaluate(
        evaluation_loader
    )

    print_metrics(
        results
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()