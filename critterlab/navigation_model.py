"""Recurrent actor for the unchanged 20-feature garden observation."""
import torch
from torch import nn

class GardenGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(20, 32, batch_first=True)
        self.head = nn.Linear(32, 4)

    def step(self, obs, hidden=None):
        features, state = self.gru(obs[:, None], None if hidden is None else hidden[None])
        return self.head(features[:, 0]), state[0]

    def sequence(self, obs, hidden=None, starts=None):
        if starts is None:
            features, _ = self.gru(obs, None if hidden is None else hidden[None])
            return self.head(features)
        state = obs.new_zeros(obs.shape[0], 32) if hidden is None else hidden
        outputs = []
        for t in range(obs.shape[1]):
            state = state * (~starts[:, t]).to(obs.dtype)[:, None]
            logits, state = self.step(obs[:, t], state)
            outputs.append(logits)
        return torch.stack(outputs, 1)

    def export(self, seed):
        return {"kind":"gru", "label":"Random-garden memory · r1", "seed":seed,
                "input_size":20, "hidden_size":32, "environment_steps":65536,
                "weight_ih":self.gru.weight_ih_l0.detach().tolist(),
                "weight_hh":self.gru.weight_hh_l0.detach().tolist(),
                "bias_ih":self.gru.bias_ih_l0.detach().tolist(),
                "bias_hh":self.gru.bias_hh_l0.detach().tolist(),
                "head":{"weight":self.head.weight.detach().tolist(),"bias":self.head.bias.detach().tolist()}}
