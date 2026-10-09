from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from torch.utils.data import DataLoader

from src.multimodal_hate.data.loaders.hateful_memes import (
    load_hateful_memes,
)

from src.multimodal_hate.training.dataset import (
    MultimodalHateSpeechDataset,
    multimodal_collate_fn,
)


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hateful_memes"
)

TRAIN_ANNOTATION_FILE = (
    DATA_ROOT / "train_weak_labels.jsonl"
)

DEV_ANNOTATION_FILE = (
    DATA_ROOT / "dev_weak_labels.jsonl"
)

TEST_ANNOTATION_FILE = (
    DATA_ROOT / "test.jsonl"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_hateful_memes_splits():
    """
    Load the Hateful Memes train/dev/test annotations.

    Training uses weakly labeled training data.

    Development uses weakly labeled development data.

    Test uses the original Hateful Memes test annotation
    because the official test split does not provide the
    required auxiliary labels.
    """

    if not TRAIN_ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            "Weakly labeled training file not found:\n"
            f"{TRAIN_ANNOTATION_FILE}\n\n"
            "Run:\n"
            "python create_weak_labels.py --split train"
        )

    if not DEV_ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            "Weakly labeled development file not found:\n"
            f"{DEV_ANNOTATION_FILE}\n\n"
            "Run:\n"
            "python create_weak_labels.py --split dev"
        )

    if not TEST_ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            "Test annotation file not found:\n"
            f"{TEST_ANNOTATION_FILE}"
        )

    train_samples = load_hateful_memes(
        str(TRAIN_ANNOTATION_FILE),
        str(DATA_ROOT),
    )

    dev_samples = load_hateful_memes(
        str(DEV_ANNOTATION_FILE),
        str(DATA_ROOT),
    )

    test_samples = load_hateful_memes(
        str(TEST_ANNOTATION_FILE),
        str(DATA_ROOT),
    )

    return (
        train_samples,
        dev_samples,
        test_samples,
    )


# ============================================================
# DATASETS
# ============================================================

def create_datasets():
    """Create lazy-loading PyTorch datasets."""

    (
        train_samples,
        dev_samples,
        test_samples,
    ) = load_hateful_memes_splits()

    train_dataset = MultimodalHateSpeechDataset(
        train_samples
    )

    dev_dataset = MultimodalHateSpeechDataset(
        dev_samples
    )

    test_dataset = MultimodalHateSpeechDataset(
        test_samples
    )

    return (
        train_dataset,
        dev_dataset,
        test_dataset,
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    batch_size: int = 4,
    num_workers: int = 0,
):
    """
    Create train/dev/test DataLoaders.

    num_workers=0 is intentional for Windows and keeps
    memory usage predictable.
    """

    (
        train_dataset,
        dev_dataset,
        test_dataset,
    ) = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=False,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=False,
    )

    return (
        train_loader,
        dev_loader,
        test_loader,
    )


# ============================================================
# SAFE SMOKE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("HATEFUL MEMES DATA PIPELINE")
    print("=" * 60)

    (
        train_loader,
        dev_loader,
        test_loader,
    ) = create_dataloaders(
        batch_size=2,
        num_workers=0,
    )

    print()
    print(
        "TRAIN SAMPLES:",
        len(train_loader.dataset),
    )

    print(
        "DEV SAMPLES:  ",
        len(dev_loader.dataset),
    )

    print(
        "TEST SAMPLES: ",
        len(test_loader.dataset),
    )

    print()
    print("Loading ONE batch only...")

    batch = next(iter(train_loader))

    print(
        "IMAGE SHAPE:       ",
        batch["image"].shape,
    )

    print(
        "TEXT COUNT:        ",
        len(batch["text"]),
    )

    print(
        "HATE TARGET SHAPE: ",
        batch["hate_target"].shape,
    )

    print(
        "SARCASM SHAPE:     ",
        batch["sarcasm_target"].shape,
    )

    print(
        "TARGET SHAPE:      ",
        batch["target_target"].shape,
    )

    # --------------------------------------------------------
    # Verify auxiliary labels are present in the batch.
    # --------------------------------------------------------

    sarcasm_positive = int(
        batch["sarcasm_target"][:, 1].sum().item()
    )

    target_positive = int(
        batch["target_target"].sum().item()
    )

    print()
    print(
        "SARCASM POSITIVE LABELS IN BATCH:",
        sarcasm_positive,
    )

    print(
        "TARGET-GROUP POSITIVE LABELS IN BATCH:",
        target_positive,
    )

    print()
    print("DATA PIPELINE TEST PASSED")
    print("=" * 60)