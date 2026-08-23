import torch
from torch import nn

from multimodal_hate.inference.predictor import (
    HateSpeechPrediction,
    MultimodalPredictor,
)


class DummyOutput:
    def __init__(self, batch_size):
        self.logits = {
            "hate": torch.tensor(
                [[-1.0, 2.0]] * batch_size
            ),
            "sarcasm": torch.tensor(
                [[2.0, -1.0]] * batch_size
            ),
            "target": torch.tensor(
                [[
                    2.0,
                    -2.0,
                    1.0,
                    -1.0,
                    3.0,
                ]] * batch_size
            ),
        }


class DummyModel(nn.Module):
    def forward(self, images, texts):
        return DummyOutput(
            images.size(0)
        )


def create_predictor():
    return MultimodalPredictor(
        model=DummyModel(),
        device="cpu",
    )


def test_predictor_returns_prediction():
    predictor = create_predictor()

    image = torch.rand(
        3,
        224,
        224,
    )

    prediction = predictor.predict(
        image=image,
        text="test meme",
        sample_id="001",
    )

    assert isinstance(
        prediction,
        HateSpeechPrediction,
    )

    assert prediction.sample_id == "001"

    assert prediction.hate_label == 1
    assert prediction.sarcasm_label == 0

    assert 0.0 <= prediction.hate_probability <= 1.0
    assert 0.0 <= prediction.sarcasm_probability <= 1.0

    assert len(
        prediction.target_probabilities
    ) == 5

    assert prediction.target_labels == [
        1,
        0,
        1,
        0,
        1,
    ]


def test_predictor_load_image(tmp_path):
    from PIL import Image

    image_path = tmp_path / "sample.png"

    image = Image.new(
        "RGB",
        (32, 32),
        (255, 255, 255),
    )

    image.save(image_path)

    predictor = create_predictor()

    tensor = predictor.load_image(
        image_path
    )

    assert tensor.shape == (
        3,
        224,
        224,
    )

    assert tensor.dtype == torch.float32

    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_predict_from_path(tmp_path):
    from PIL import Image

    image_path = tmp_path / "sample.png"

    image = Image.new(
        "RGB",
        (32, 32),
        (255, 255, 255),
    )

    image.save(image_path)

    predictor = create_predictor()

    prediction = predictor.predict_from_path(
        image_path=image_path,
        text="test meme",
        sample_id="002",
    )

    assert prediction.sample_id == "002"
    assert prediction.hate_label == 1


def test_predict_batch():
    predictor = create_predictor()

    images = torch.rand(
        2,
        3,
        224,
        224,
    )

    texts = [
        "first meme",
        "second meme",
    ]

    predictions = predictor.predict_batch(
        images=images,
        texts=texts,
        sample_ids=[
            "001",
            "002",
        ],
    )

    assert len(predictions) == 2

    assert predictions[0].sample_id == "001"
    assert predictions[1].sample_id == "002"

    assert predictions[0].hate_label == 1
    assert predictions[1].hate_label == 1


def test_predict_batch_validates_text_count():
    predictor = create_predictor()

    images = torch.rand(
        2,
        3,
        224,
        224,
    )

    try:
        predictor.predict_batch(
            images=images,
            texts=["only one"],
        )
    except ValueError as exc:
        assert "texts" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError."
        )