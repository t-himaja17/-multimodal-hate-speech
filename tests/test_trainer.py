import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from multimodal_hate.training.trainer import MultimodalTrainer


class DummyMultimodalModel(nn.Module):
    """
    Lightweight model used only to test the training pipeline.

    It mimics the output interface expected by MultimodalTrainer.
    """

    def __init__(self):
        super().__init__()

        self.image_projection = nn.Linear(
            3 * 8 * 8,
            512,
        )

        self.text_projection = nn.Linear(
            16,
            512,
        )

        self.hate_head = nn.Linear(
            512,
            2,
        )

        self.sarcasm_head = nn.Linear(
            512,
            2,
        )

        self.target_head = nn.Linear(
            512,
            5,
        )

    def forward(
        self,
        images,
        texts,
    ):
        batch_size = images.size(0)

        image_features = images.reshape(
            batch_size,
            -1,
        )

        image_representation = self.image_projection(
            image_features
        )

        # Create deterministic lightweight text features.
        text_features = torch.zeros(
            batch_size,
            16,
            device=images.device,
        )

        text_representation = self.text_projection(
            text_features
        )

        pooled = (
            image_representation
            + text_representation
        ) / 2.0

        hate_logits = self.hate_head(
            pooled
        )

        sarcasm_logits = self.sarcasm_head(
            pooled
        )

        target_logits = self.target_head(
            pooled
        )

        return type(
            "DummyOutput",
            (),
            {
                "logits": {
                    "hate": hate_logits,
                    "sarcasm": sarcasm_logits,
                    "target": target_logits,
                },
                "image_representation": (
                    image_representation
                ),
                "text_representation": (
                    text_representation
                ),
            },
        )()


def create_batch(batch_size=4):
    """Create a small synthetic multimodal batch."""

    images = torch.randn(
        batch_size,
        3,
        8,
        8,
    )

    texts = [
        f"sample {index}"
        for index in range(batch_size)
    ]

    hate_targets = torch.zeros(
        batch_size,
        2,
    )

    sarcasm_targets = torch.zeros(
        batch_size,
        2,
    )

    target_targets = torch.zeros(
        batch_size,
        5,
    )

    for index in range(batch_size):
        hate_targets[index, index % 2] = 1.0
        sarcasm_targets[index, (index + 1) % 2] = 1.0
        target_targets[
            index,
            index % 5,
        ] = 1.0

    return {
        "image": images,
        "text": texts,
        "hate_target": hate_targets,
        "sarcasm_target": sarcasm_targets,
        "target_target": target_targets,
        "sample_id": [
            str(index)
            for index in range(batch_size)
        ],
    }


class SyntheticDataset(torch.utils.data.Dataset):
    """Small dataset for trainer integration testing."""

    def __init__(self):
        self.samples = [
            create_batch(1)
            for _ in range(8)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch):
    return {
        "image": torch.cat(
            [
                item["image"]
                for item in batch
            ],
            dim=0,
        ),
        "text": [
            text
            for item in batch
            for text in item["text"]
        ],
        "hate_target": torch.cat(
            [
                item["hate_target"]
                for item in batch
            ],
            dim=0,
        ),
        "sarcasm_target": torch.cat(
            [
                item["sarcasm_target"]
                for item in batch
            ],
            dim=0,
        ),
        "target_target": torch.cat(
            [
                item["target_target"]
                for item in batch
            ],
            dim=0,
        ),
        "sample_id": [
            sample_id
            for item in batch
            for sample_id in item["sample_id"]
        ],
    }


def test_trainer_runs_one_epoch(tmp_path):
    torch.manual_seed(42)

    model = DummyMultimodalModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    dataset = SyntheticDataset()

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        gradient_accumulation_steps=1,
        mixed_precision=False,
        checkpoint_dir=tmp_path,
    )

    result = trainer.train_epoch(
        loader
    )

    assert result.batches == 2
    assert result.loss >= 0.0
    assert result.hate_loss >= 0.0
    assert result.sarcasm_loss >= 0.0
    assert result.contrastive_loss >= 0.0
    assert result.target_loss >= 0.0


def test_trainer_validation_runs(tmp_path):
    torch.manual_seed(42)

    model = DummyMultimodalModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    dataset = SyntheticDataset()

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=tmp_path,
    )

    result = trainer.validate_epoch(
        loader
    )

    assert result.batches == 2
    assert result.loss >= 0.0


def test_trainer_saves_checkpoint(tmp_path):
    torch.manual_seed(42)

    model = DummyMultimodalModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    dataset = SyntheticDataset()

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=tmp_path,
    )

    trainer.fit(
        train_loader=loader,
        validation_loader=loader,
        epochs=1,
    )

    assert (
        tmp_path / "latest.pt"
    ).is_file()

    assert (
        tmp_path / "best.pt"
    ).is_file()