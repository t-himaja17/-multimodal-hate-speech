from pathlib import Path

from torch.utils.data import DataLoader, Subset

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

DATA_ROOT = Path("data/raw/hateful_memes")


# ============================================================
# LOAD DATA
# ============================================================

def load_hateful_memes_splits():
    """Load the Hateful Memes train/dev/test annotations."""

    train_samples = load_hateful_memes(
        str(DATA_ROOT / "train.jsonl"),
        str(DATA_ROOT),
    )

    dev_samples = load_hateful_memes(
        str(DATA_ROOT / "dev.jsonl"),
        str(DATA_ROOT),
    )

    test_samples = load_hateful_memes(
        str(DATA_ROOT / "test.jsonl"),
        str(DATA_ROOT),
    )

    return train_samples, dev_samples, test_samples


# ============================================================
# DATASETS
# ============================================================

def create_datasets():
    """Create lazy-loading PyTorch datasets."""

    train_samples, dev_samples, test_samples = (
        load_hateful_memes_splits()
    )

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

    num_workers=0 is intentional for the first Windows
    smoke test. It keeps memory usage predictable.
    """

    train_dataset, dev_dataset, test_dataset = (
        create_datasets()
    )

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

    train_loader, dev_loader, test_loader = (
        create_dataloaders(
            batch_size=2,
            num_workers=0,
        )
    )

    print()
    print("TRAIN SAMPLES:", len(train_loader.dataset))
    print("DEV SAMPLES:  ", len(dev_loader.dataset))
    print("TEST SAMPLES: ", len(test_loader.dataset))

    print()
    print("Loading ONE batch only...")

    batch = next(iter(train_loader))

    print("IMAGE SHAPE:       ", batch["image"].shape)
    print("TEXT COUNT:        ", len(batch["text"]))
    print("HATE TARGET SHAPE: ", batch["hate_target"].shape)
    print("SARCASM SHAPE:     ", batch["sarcasm_target"].shape)
    print("TARGET SHAPE:      ", batch["target_target"].shape)

    print()
    print("DATA PIPELINE TEST PASSED")
    print("=" * 60)
