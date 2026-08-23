import json

from src.multimodal_hate.data.loaders.hateful_memes import load_hateful_memes


def test_load_hateful_memes(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    (image_dir / "123.png").write_bytes(b"fake image")

    annotation_file = tmp_path / "train.jsonl"

    annotation = {
        "id": "123",
        "img": "img/123.png",
        "text": "example text",
        "label": 1,
        "target_group": "example_group",
    }

    with annotation_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(annotation) + "\n")

    samples = load_hateful_memes(
        str(annotation_file),
        str(image_dir),
    )

    assert len(samples) == 1

    sample = samples[0]

    assert sample.sample_id == "123"
    assert sample.source == "hateful_memes"
    assert sample.text == "example text"
    assert sample.hate_label == 1
    assert sample.image_path is not None
    assert sample.metadata["target_group"] == "example_group"