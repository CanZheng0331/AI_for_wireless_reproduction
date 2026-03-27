import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, m: int, n: int, norm_type: str = "fixed"):
        super().__init__()
        self.m = m
        self.norm_type = norm_type
        self.tx_dense_relu = nn.Linear(m, m)
        self.tx_dense_linear = nn.Linear(m, n)
        self.rx_dense_relu = nn.Linear(n, m)
        self.rx_dense_linear = nn.Linear(m, m)
        self.n = n

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = F.relu(self.tx_dense_relu(x))
        encoded = self.tx_dense_linear(encoded)
        if self.norm_type == "fixed":
            norm = torch.norm(encoded, p=2, dim=1, keepdim=True)
            return math.sqrt(self.n) * (encoded / torch.clamp(norm, min=1e-8))
        if self.norm_type == "average":
            mean_energy = torch.mean(torch.sum(encoded**2, dim=1, keepdim=True))
            return math.sqrt(self.n) * (encoded / torch.sqrt(torch.clamp(mean_energy, min=1e-8)))
        return encoded

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        decoded = F.relu(self.rx_dense_relu(y))
        return self.rx_dense_linear(decoded)

    def forward(self, x: torch.Tensor, noise_std: float) -> torch.Tensor:
        tx = self.encode(x)
        if self.training:
            tx = tx + torch.randn_like(tx) * noise_std
        return self.decode(tx)
