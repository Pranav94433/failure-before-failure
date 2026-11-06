"""PyTorch network architecture shared by training and inference."""

from __future__ import annotations

import torch
from torch import nn


class Autoencoder(nn.Module):
    """A small fully connected network that reconstructs sensor feature rows."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))
