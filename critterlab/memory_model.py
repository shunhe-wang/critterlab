"""Matched-data reactive and recurrent actors, with portable real weights."""
import torch
from torch import nn


def dense(layer):
    return {"weight": layer.weight.detach().tolist(), "bias": layer.bias.detach().tolist()}


class Reactive(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(22, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)

    def step(self, x, hidden=None):
        return self(x), None

    def export(self, seed):
        return {"kind": "reactive", "label": "Memory-task reactive", "seed": seed,
                "architecture": [22, 64, 64, 4], "activation": "tanh", "environment_steps": 0,
                "layers": [dense(layer) for layer in self.net if isinstance(layer, nn.Linear)]}


class Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(22, 32, batch_first=True)
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        features, _ = self.gru(x)
        return self.head(features)

    def step(self, x, hidden=None):
        features, state = self.gru(x[:, None, :], None if hidden is None else hidden[None, :, :])
        return self.head(features[:, 0, :]), state[0]

    def export(self, seed):
        return {"kind": "gru", "label": "Vanishing-clue GRU", "seed": seed, "input_size": 22, "hidden_size": 32,
                "weight_ih": self.gru.weight_ih_l0.detach().tolist(), "weight_hh": self.gru.weight_hh_l0.detach().tolist(),
                "bias_ih": self.gru.bias_ih_l0.detach().tolist(), "bias_hh": self.gru.bias_hh_l0.detach().tolist(), "head": dense(self.head)}
