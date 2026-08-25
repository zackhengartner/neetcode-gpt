import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        result = []
        with torch.no_grad():
            out = x
            for layer in model:
                out = layer(out)

                if isinstance(layer,nn.Linear):
                    result.append({
                        "mean": round(torch.mean(out).item(),4),
                        "std": round(torch.std(out).item(),4),
                        "dead_fraction": round((out <= 0).all(dim=0).float().mean().item(),4)
                    })
        return result
            

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()

        predictions = model(x)

        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions,y)

        loss.backward()

        result = []

        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                result.append({
                        "mean": round(grad.mean().item(), 4),
                        "std": round(grad.std().item(), 4),
                        "norm": round(grad.norm().item(), 4)
                })
        return result

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        for activation_stat in activation_stats:
            if activation_stat["dead_fraction"] > 0.5:
                return "dead_neurons"
            if activation_stat["std"] < 0.1:
                return "vanishing_gradients"
            elif activation_stat["std"] > 10.0:
                return "exploding_gradients"
        for gradient_stat in gradient_stats:
            if gradient_stat["norm"] > 1000:
                return "exploding_gradients"
            elif gradient_stat["norm"] < 1e-5:
                return "vanishing_gradients"
        return "healthy"
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        
