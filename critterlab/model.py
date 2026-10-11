"""Tiny PyTorch actor. JSON exports use the same dense/tanh math in the browser."""
import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))

    def forward(self, observation):
        return self.net(observation)

    def export(self, label: str, seed: int, steps: int):
        layers = []
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layers.append({"weight": layer.weight.detach().cpu().tolist(),
                               "bias": layer.bias.detach().cpu().tolist()})
        return {"label": label, "seed": seed, "environment_steps": steps,
                "architecture": [20, 64, 64, 4], "activation": "tanh", "layers": layers}
