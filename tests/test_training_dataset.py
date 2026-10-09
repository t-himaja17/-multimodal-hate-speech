import torch
from PIL import Image

from src.multimodal_hate.data.schema import MultimodalSample
from src.multimodal_hate.training.dataset import (
    MultimodalHateSpeechDataset,
    multimodal_collate_fn,
)


def create_image(path):
    image = Image.new(
        "RGB",
        (32, 32),
        (255, 255, 255),
    )
    image.save(path)


def test_dataset_returns_expected_shapes(tmp_path):
    image_path = tmp_path / "sample.png"
    create_image(image_path)

    sample = MultimodalSample(
        sample_id="1",
        source="test",
        image_path=str(image_path),
        text="test meme",
        hate_label=1,
        sarcasm_label=0,
        target_group="race",
    )

    dataset = MultimodalHateSpeechDataset(
        [sample]
    )

    item = dataset[0]

    assert item["image"].shape == (3, 224, 224)
    assert item["text"] == "test meme"
    assert item["hate_target"].shape == (2,)
    assert item["sarcasm_target"].shape == (2,)
    assert item["target_target"].shape == (5,)


def test_binary_targets_are_one_hot(tmp_path):
    image_path = tmp_path / "sample.png"
    create_image(image_path)

    sample = MultimodalSample(
        sample_id="1",
        source="test",
        image_path=str(image_path),
        text="test",
        hate_label=1,
        sarcasm_label=0,
    )

    dataset = MultimodalHateSpeechDataset(
        [sample]
    )

    item = dataset[0]

    assert torch.equal(
        item["hate_target"],
        torch.tensor([0.0, 1.0]),
    )

    assert torch.equal(
        item["sarcasm_target"],
        torch.tensor([1.0, 0.0]),
    )


def test_target_group_vector(tmp_path):
    image_path = tmp_path / "sample.png"
    create_image(image_path)

    sample = MultimodalSample(
        sample_id="1",
        source="test",
        image_path=str(image_path),
        text="test",
        target_group="gender",
    )

    dataset = MultimodalHateSpeechDataset(
        [sample]
    )

    item = dataset[0]

    assert torch.equal(
        item["target_target"],
        torch.tensor(
            [0.0, 0.0, 1.0, 0.0, 0.0]
        ),
    )


def test_collate_function(tmp_path):
    image_1 = tmp_path / "1.png"
    image_2 = tmp_path / "2.png"

    create_image(image_1)
    create_image(image_2)

    samples = [
        MultimodalSample(
            sample_id="1",
            source="test",
            image_path=str(image_1),
            text="first",
            hate_label=1,
            sarcasm_label=0,
            target_group="race",
        ),
        MultimodalSample(
            sample_id="2",
            source="test",
            image_path=str(image_2),
            text="second",
            hate_label=0,
            sarcasm_label=1,
            target_group="religion",
        ),
    ]

    dataset = MultimodalHateSpeechDataset(
        samples
    )

    batch = multimodal_collate_fn(
        [
            dataset[0],
            dataset[1],
        ]
    )

    assert batch["image"].shape == (
        2,
        3,
        224,
        224,
    )

    assert batch["hate_target"].shape == (
        2,
        2,
    )

    assert batch["sarcasm_target"].shape == (
        2,
        2,
    )

    assert batch["target_target"].shape == (
        2,
        5,
    )

    assert batch["text"] == [
        "first",
        "second",
    ]