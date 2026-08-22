from transformers import AutoModel, AutoTokenizer

from .base import BaseEncoder


class RoBERTaEncoder(BaseEncoder):
    """RoBERTa-large encoder for additional text representations."""

    def __init__(
        self,
        model_name: str = "FacebookAI/roberta-large",
        max_length: int = 128,
    ):
        super().__init__(output_dim=1024)

        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, text):
        """Encode text and return the [CLS] representation."""

        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.model.device)
            for key, value in encoded.items()
        }

        outputs = self.model(**encoded)

        return outputs.last_hidden_state[:, 0, :]