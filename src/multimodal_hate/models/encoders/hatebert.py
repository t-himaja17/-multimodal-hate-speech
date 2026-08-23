import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaseEncoder


class HateBERTEncoder(BaseEncoder):
    """HateBERT encoder for structured OCR text and metadata.

    The default forward() API remains backward-compatible and returns
    the [CLS] representation with shape [B, 768].

    encode_tokens() exposes the complete token sequence with shape
    [B, N_txt, 768] for multimodal transformer fusion.
    """

    def __init__(
        self,
        model_name: str = "GroNLP/hateBERT",
        max_length: int = 128,
    ):
        super().__init__(output_dim=768)

        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _tokenize(self, text):
        """Tokenize text and move tensors to the model device."""

        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device

        return {
            key: value.to(device)
            for key, value in encoded.items()
        }

    def forward(self, text) -> torch.Tensor:
        """Return the [CLS] representation.

        Returns:
            Tensor of shape [B, 768].
        """

        encoded = self._tokenize(text)

        outputs = self.model(**encoded)

        return outputs.last_hidden_state[:, 0, :]

    def encode_tokens(self, text) -> torch.Tensor:
        """Return the complete HateBERT token sequence.

        This interface is used by the multimodal fusion transformer.

        Returns:
            Tensor of shape [B, N_txt, 768].
        """

        encoded = self._tokenize(text)

        outputs = self.model(**encoded)

        return outputs.last_hidden_state