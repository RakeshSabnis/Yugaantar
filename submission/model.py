"""Policy network: ray CNN + entity attention + vitals MLP -> GRU memory -> action / value / helper heads.

Used unchanged by training (training/) and by the submitted agent (submission/agent.py).
Inputs are only the official observation fields plus the agent's own previous action.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actions import NUM_ACTIONS  # noqa: E402

NUM_RAYS = 64
RAY_TYPES = 6
MAX_ENT = 32
ENT_FEAT = 5
ENT_TYPES = 12
VITALS = 17
AUX_GOALS = 5  # teacher goal kind: none, exit, key, frontier, revisit

# Flat feature layout (float32), produced by encode_obs and consumed by Policy.
RAY_DIM = NUM_RAYS * (1 + RAY_TYPES)
ENT_DIM = MAX_ENT * (ENT_FEAT + 2)  # features, type id, mask
OBS_DIM = RAY_DIM + ENT_DIM + VITALS
VITAL_SCALE = np.array([100, 100, 7, 50, 100, 50, 25, 100, 1, 1, 1, 1, 1, 1, 10, 10, 10], np.float32)


def encode_obs(obs: dict) -> np.ndarray:
    """Observation dict -> flat float32 vector of length OBS_DIM. NaN/inf-safe."""
    out = np.zeros(OBS_DIM, np.float32)
    rays = obs["rays"]
    dist = np.asarray(rays["distance"], np.float32).reshape(NUM_RAYS)
    hit = np.clip(np.asarray(rays["hit_type"], np.int64).reshape(NUM_RAYS), 0, RAY_TYPES - 1)
    out[:NUM_RAYS] = dist
    onehot = np.zeros((RAY_TYPES, NUM_RAYS), np.float32)
    onehot[hit, np.arange(NUM_RAYS)] = 1.0
    out[NUM_RAYS:RAY_DIM] = onehot.reshape(-1)
    ents = obs["visible_entities"]
    e = np.zeros((MAX_ENT, ENT_FEAT + 2), np.float32)
    e[:, :ENT_FEAT] = np.asarray(ents["features"], np.float32).reshape(MAX_ENT, ENT_FEAT)
    e[:, ENT_FEAT] = np.clip(np.asarray(ents["entity_type"], np.int64).reshape(MAX_ENT), 0, ENT_TYPES - 1)
    e[:, ENT_FEAT + 1] = np.asarray(ents["mask"], np.float32).reshape(MAX_ENT)
    e[:, 3:5] = np.clip(e[:, 3:5] / 10.0, -5, 5)  # relative velocities
    out[RAY_DIM:RAY_DIM + ENT_DIM] = e.reshape(-1)
    out[RAY_DIM + ENT_DIM:] = np.clip(np.asarray(obs["vitals"], np.float32).reshape(VITALS) / VITAL_SCALE, -5, 5)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    return out


class Policy(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        # Rays: circular 1D CNN over 64 rays x (distance, near-ness, 6 hit types)
        self.ray_conv = nn.Sequential(
            nn.Conv1d(2 + RAY_TYPES, 32, 5, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),
        )
        self.ray_fc = nn.Linear(64 * 16, 256)
        # Entities: per-entity MLP, then attention pooling (learned query) + max pooling, masked
        self.ent_type = nn.Embedding(ENT_TYPES, 16)
        self.ent_mlp = nn.Sequential(nn.Linear(ENT_FEAT + 1 + 16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ent_query = nn.Parameter(torch.randn(2, 64) * 0.1)
        # Vitals + previous action
        self.vit_mlp = nn.Sequential(nn.Linear(VITALS + NUM_ACTIONS, 64), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(256 + 64 * 3 + 64, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.pi = nn.Sequential(nn.Linear(hidden * 2, 256), nn.ReLU(), nn.Linear(256, NUM_ACTIONS))
        self.v = nn.Sequential(nn.Linear(hidden * 2, 256), nn.ReLU(), nn.Linear(256, 1))
        self.aux_goal = nn.Linear(hidden, AUX_GOALS)       # what the teacher is heading for
        self.aux_locked = nn.Linear(hidden, 1)              # has a locked door been seen this episode?

    def initial_state(self, batch: int = 1, device="cpu") -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden, device=device)

    def encode(self, x: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        """x: (N, OBS_DIM) float, prev_action: (N,) long -> (N, hidden)."""
        n = x.shape[0]
        dist = x[:, :NUM_RAYS]
        hit = x[:, NUM_RAYS:RAY_DIM].view(n, RAY_TYPES, NUM_RAYS)
        rays = torch.cat([dist.unsqueeze(1), (1.0 - dist).pow(4).unsqueeze(1), hit], dim=1)
        r = self.ray_conv(rays).flatten(1)
        r = Fn.relu(self.ray_fc(r))
        e = x[:, RAY_DIM:RAY_DIM + ENT_DIM].view(n, MAX_ENT, ENT_FEAT + 2)
        mask = e[:, :, ENT_FEAT + 1] > 0.5
        etype = self.ent_type(e[:, :, ENT_FEAT].long().clamp(0, ENT_TYPES - 1))
        feats = torch.cat([e[:, :, :ENT_FEAT], (1.0 - e[:, :, :1]).clamp(0, 1), etype], dim=-1)
        h = self.ent_mlp(feats)  # (n, M, 64)
        scores = torch.einsum("nmd,qd->nqm", h, self.ent_query) / 8.0
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        att = torch.softmax(scores, dim=-1) * mask.unsqueeze(1).float()
        pooled = torch.einsum("nqm,nmd->nqd", att, h).flatten(1)  # (n, 128)
        mx = h.masked_fill(~mask.unsqueeze(-1), 0.0).max(dim=1).values
        vit = x[:, RAY_DIM + ENT_DIM:]
        pa = Fn.one_hot(prev_action.clamp(0, NUM_ACTIONS - 1), NUM_ACTIONS).float()
        v = self.vit_mlp(torch.cat([vit, pa], dim=-1))
        return self.fuse(torch.cat([r, pooled, mx, v], dim=-1))

    def forward(self, x, prev_action, h0):
        """x: (B, T, OBS_DIM), prev_action: (B, T), h0: (1, B, H) -> logits (B,T,A), value (B,T), aux, hT."""
        b, t, _ = x.shape
        z = self.encode(x.reshape(b * t, -1), prev_action.reshape(b * t)).view(b, t, -1)
        out, hT = self.gru(z, h0)
        feat = torch.cat([out, z], dim=-1)
        return self.pi(feat), self.v(feat).squeeze(-1), (self.aux_goal(out), self.aux_locked(out).squeeze(-1)), hT

    @torch.no_grad()
    def step(self, x: torch.Tensor, prev_action: torch.Tensor, h: torch.Tensor):
        """Single step for acting. x: (N, OBS_DIM) -> logits (N, A), value (N,), h."""
        logits, value, _, h = self.forward(x.unsqueeze(1), prev_action.unsqueeze(1), h)
        return logits[:, 0], value[:, 0], h
