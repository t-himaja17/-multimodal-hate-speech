import json

from src.multimodal_hate.data.loaders.multioff import load_multioff


def test_load_multioff(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    (image_dir / "001.png").touch()
    (image_dir / "002.png").touch()

    annotation_file = tmp_path / "train.jsonl"

    records = [
        {
            "id": "001",
            "img": "001.png",
            "text": "This is a test meme.",
            "label": 1,
            "target_group": "example_group",
        },
        {
            "id": "002",
            "img": "002.png",
            "text": "Another test meme.",
            "label": 0,
        },
    ]

    with annotation_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    samples = load_multioff(
        str(annotation_file),
        str(image_dir),
    )

    assert len(samples) == 2

    assert samples[0].sample_id == "001"
    assert samples[0].source == "multioff"
    assert samples[0].text == "This is a test meme."
    assert samples[0].hate_label == 1
    assert samples[0].image_path is not None
    assert samples[0].metadata["target_group"] == "example_group"

    assert samples[1].sample_id == "002"
    assert samples[1].hate_label == 0