import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.train_mmsd_sarcasm import (
    MMSDSarcasmDataset,
    collate_batch,
    create_model,
    create_optimizer,
)
from src.multimodal_hate.training.trainer import MultimodalTrainer
from src.multimodal_hate.training.losses import MultimodalTotalLoss

SOURCE_DIR = PROJECT_ROOT / "artifacts" / "checkpoints_mmsd_sarcasm"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints_mmsd_sarcasm_resume"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    latest = SOURCE_DIR / "latest.pt"
    best = SOURCE_DIR / "best.pt"

    source_checkpoint = latest if latest.is_file() else best

    if not source_checkpoint.is_file():
        raise FileNotFoundError(
            f"No checkpoint found in {SOURCE_DIR}"
        )

    print("Resume source:", source_checkpoint)
    print("New checkpoints:", OUTPUT_DIR)

    train_dataset = MMSDSarcasmDataset("train")
    validation_dataset = MMSDSarcasmDataset("validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=OUTPUT_DIR,
        mixed_precision=(device.type == "cuda"),
        gradient_clipping=1.0,
    )

    checkpoint = trainer.load_checkpoint(source_checkpoint)

    start_epoch = int(checkpoint.get("epoch", 0))
    previous_best = float(
        checkpoint.get("best_validation_loss", float("inf"))
    )

    print("Checkpoint epoch:", start_epoch)
    print("Previous best validation loss:", previous_best)

    print(
        "\nWARNING: this uses the existing trainer's fit() method, "
        "which resets its early-stopping baseline and history."
    )
    print(
        "This run is isolated in a new directory, so the original "
        "sarcasm checkpoints will not be overwritten."
    )

    trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=3,
        early_stopping_patience=2,
    )

    print("Resume run finished.")
    print("Checkpoints saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()