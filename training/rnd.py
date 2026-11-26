"""Learned curiosity: random network distillation (Burda et al., 2018), over spatial inputs.

A fixed, randomly initialised target network maps each state to a small vector. A predictor network is
trained to match it on the states the agent visits. States the agent has rarely been in are predicted
badly, so the prediction error is used as an intrinsic "novelty" reward.

Input: the encoded observation (rays, entities, vitals) plus the zoomed-out view of the agent's own SLAM
map, so novelty is about where the agent is and what it sees, not about noise.
Used in training only (training/dagger.py); the submitted agent does not contain or need it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
from model import MAP_DIM, OBS_DIM, WIDE_DIM  # noqa: E402

IN_DIM = OBS_DIM + WIDE_DIM * 8  # observation + 8 bit-planes of the wide map view


def rnd_input(x: torch.Tensor) -> torch.Tensor:
    """x: (N, FULL_DIM) network input -> (N, IN_DIM)."""
    wide = x[:, OBS_DIM + MAP_DIM:OBS_DIM + MAP_DIM + WIDE_DIM].round().long()
    bits = torch.stack([(wide >> b) & 1 for b in range(8)], dim=-1).float().flatten(1)
    return torch.cat([x[:, :OBS_DIM], bits], dim=-1)


class RND(nn.Module):
    def __init__(self, out: int = 32):
        super().__init__()
        self.target = nn.Sequential(nn.Linear(IN_DIM, 256), nn.ReLU(), nn.Linear(256, out))
        self.predictor = nn.Sequential(nn.Linear(IN_DIM, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, out))
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.register_buffer("err_mean", torch.tensor(1.0))  # running mean of the error, for scaling

    def error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-state prediction error (N,). Gradients flow into the predictor only."""
        z = rnd_input(x)
        return ((self.predictor(z) - self.target(z)) ** 2).mean(dim=-1)

    @torch.no_grad()
    def novelty(self, x: torch.Tensor) -> torch.Tensor:
        """Intrinsic reward in [0, 3]: error relative to its running mean."""
        return (self.error(x) / self.err_mean.clamp_min(1e-6)).clamp(0.0, 3.0)
