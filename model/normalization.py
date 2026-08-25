import numpy as np
from numpy.typing import NDArray
import torch


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        x_t = torch.as_tensor(x, dtype=torch.float)
        gamma_t = torch.as_tensor(gamma, dtype=torch.float)
        beta_t =  torch.as_tensor(beta, dtype=torch.float)
        mean = torch.mean(x_t)
        variance = torch.mean((x_t - mean) ** 2)
        final_t = ((x_t - mean) / torch.sqrt(variance + 1e-5)) * gamma_t + beta_t
        return np.round(final_t.numpy(), 5)