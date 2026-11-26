"""Policy network and observation encoding shared by training and the evaluation agent.

Four encoders feed one recurrent core:
  rays      64 x (distance + hit-type one-hot)  -> circular 1D CNN (rays wrap around 360 degrees)
  entities  up to 32 x (5 features + class)      -> shared MLP + attention pooling (any number of enemies)
  memory    2 egocentric crops of the map built by submission/mapper.py -> small 2D CNN
  body      vitals + previous action             -> MLP
  fused -> GRU -> 9 independent button logits (+ a value head used by PPO fine-tuning)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mapper import CROP, NUM_LAYERS

NUM_BUTTONS = 9
RAY_FEATS = 7
ENT_FEATS = 5 + 12
BODY_FEATS = 17 + NUM_BUTTONS
MAP_CH = 2 * NUM_LAYERS
_VITAL_SCALE = np.array([100, 100, 7, 50, 50, 20, 10, 100, 1, 1, 1, 1, 1, 1, 8, 8, 8], np.float32)
_EYE6 = np.eye(6, dtype=np.float32)
_EYE12 = np.eye(12, dtype=np.float32)


def encode(obs: dict, crops: np.ndarray, prev_action: np.ndarray) -> dict:
    """Observation -> fixed-size float arrays (no decisions are made here)."""
    rays = np.concatenate([np.asarray(obs["rays"]["distance"], np.float32)[:, None], _EYE6[np.asarray(obs["rays"]["hit_type"])]], axis=1)
    ents = obs["visible_entities"]
    mask = np.asarray(ents["mask"]).astype(bool)
    f = np.asarray(ents["features"], np.float32).copy()
    f[:, 3:5] /= 8.0
    f[:, 0] = np.minimum(f[:, 0], 4.0)
    ent = np.concatenate([f, _EYE12[np.asarray(ents["entity_type"])]], axis=1) * mask[:, None]
    body = np.concatenate([np.clip(np.asarray(obs["vitals"], np.float32) / _VITAL_SCALE, -4, 4), np.asarray(prev_action, np.float32)])
    return {"rays": rays, "ent": ent.astype(np.float32), "ent_mask": mask, "body": body, "map": crops}


class Policy(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.ray = nn.Sequential(
            nn.Conv1d(RAY_FEATS, 32, 5, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(48, 64, 3, stride=2, padding=1, padding_mode="circular"), nn.ReLU(),
        )
        self.ray_out = nn.Linear(64 * 16, 128)
        self.ent_in = nn.Sequential(nn.Linear(ENT_FEATS, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ent_query = nn.Parameter(torch.randn(4, 64) * 0.1)  # 4 learned queries attend over the entity set
        self.ent_out = nn.Linear(4 * 64, 96)
        self.map = nn.Sequential(
            nn.Conv2d(MAP_CH, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.map_out = nn.Linear(64 * (CROP // 8) ** 2, 192)
        self.body = nn.Sequential(nn.Linear(BODY_FEATS, 64), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(128 + 96 + 192 + 64, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.pi = nn.Linear(hidden, NUM_BUTTONS)
        self.v = nn.Linear(hidden, 1)

    def features(self, x: dict) -> torch.Tensor:
        """x: tensors shaped (N, ...). Returns (N, hidden)."""
        r = self.ray(x["rays"].transpose(1, 2)).flatten(1)
        r = F.relu(self.ray_out(r))
        e = self.ent_in(x["ent"])  # (N, 32, 64)
        att = torch.einsum("qd,nkd->nqk", self.ent_query, e) / 8.0
        att = att.masked_fill(~x["ent_mask"][:, None, :], -1e9).softmax(-1)
        any_ent = x["ent_mask"].any(1)[:, None, None]
        pooled = torch.einsum("nqk,nkd->nqd", att * any_ent, e).flatten(1)
        e = F.relu(self.ent_out(pooled))
        m = F.relu(self.map_out(self.map(x["map"]).flatten(1)))
        b = self.body(x["body"])
        return self.fuse(torch.cat([r, e, m, b], 1))

    def forward(self, x: dict, h: torch.Tensor | None = None):
        """x: tensors shaped (B, T, ...). Returns logits (B, T, 9), value (B, T), hidden (1, B, H)."""
        B, T = x["body"].shape[:2]
        z = self.features({k: v.flatten(0, 1) for k, v in x.items()}).view(B, T, -1)
        out, h = self.gru(z, h)
        return self.pi(out), self.v(out).squeeze(-1), h
