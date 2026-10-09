import pytest
import torch

from multimodal_hate.training.losses import (
    BinaryClassificationLoss,
    MultiLabelBCELoss,
    NTXentLoss,
    MultimodalTotalLoss,
)


def test_binary_classification_loss():
    loss_fn = BinaryClassificationLoss()

    logits = torch.tensor(
        [
            [2.0, -2.0],
            [-2.0, 2.0],
        ]
    )

    targets = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    loss = loss_fn(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_multilabel_bce_loss():
    loss_fn = MultiLabelBCELoss()

    logits = torch.randn(4, 5)
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    loss = loss_fn(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_nt_xent_loss():
    loss_fn = NTXentLoss(
        temperature=0.07,
    )

    image_representation = torch.randn(4, 512)
    text_representation = torch.randn(4, 512)

    loss = loss_fn(
        image_representation,
        text_representation,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_nt_xent_requires_batch_size_two():
    loss_fn = NTXentLoss()

    image_representation = torch.randn(1, 512)
    text_representation = torch.randn(1, 512)

    with pytest.raises(ValueError):
        loss_fn(
            image_representation,
            text_representation,
        )


def test_total_loss():
    loss_fn = MultimodalTotalLoss()

    hate_logits = torch.randn(4, 2)
    hate_targets = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    sarcasm_logits = torch.randn(4, 2)
    sarcasm_targets = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    target_logits = torch.randn(4, 5)
    target_targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )

    image_representation = torch.randn(4, 512)
    text_representation = torch.randn(4, 512)

    losses = loss_fn(
        hate_logits=hate_logits,
        hate_targets=hate_targets,
        sarcasm_logits=sarcasm_logits,
        sarcasm_targets=sarcasm_targets,
        target_logits=target_logits,
        target_targets=target_targets,
        image_representation=image_representation,
        text_representation=text_representation,
    )

    assert set(losses.keys()) == {
        "hate",
        "sarcasm",
        "contrastive",
        "target",
        "total",
    }

    for value in losses.values():
        assert value.ndim == 0
        assert torch.isfinite(value)

    expected_total = (
        losses["hate"]
        + 0.3 * losses["sarcasm"]
        + 0.1 * losses["contrastive"]
        + 0.2 * losses["target"]
    )

    assert torch.allclose(
        losses["total"],
        expected_total,
    )