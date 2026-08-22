from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Common interface for all multimodal encoders."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Encode the input and return its representation."""
        raise NotImplementedError