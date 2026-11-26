"""Evaluation entry point (docs/CONTRACT.md section 6).

Every action comes from the trained network in model.py. This file only does what the rules allow
around it: observation preprocessing (the spatial memory in mapper.py), action masking from the
agent's own previous action, and a safety net so a fault can never turn into a crash or an invalid
action (which would leave the player standing still for the rest of the map).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS = SUBMISSION_DIR / "weights" / "policy.pt"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

FWD, BACK, LEFT, RIGHT, TL, TR, ATK, USE, NEXTW = range(9)


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.n = int(action_space.shape[0]) if action_space is not None else 9
        self.ok = False
        self.rng = np.random.default_rng(0)
        self.prev = np.zeros(self.n, np.int8)
        self.t = 0
        try:
            import torch
            from mapper import Mapper
            from model import Policy, encode

            torch.set_num_threads(1)
            self.torch, self.encode = torch, encode
            self.mapper = Mapper()
            # The model is ~1M parameters: one CPU thread runs it in a few ms, well inside the 40 ms
            # budget, so we always run on CPU and never touch the shared GPU.
            self.device = "cpu"
            self.net = Policy()
            if WEIGHTS.is_file():
                state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
                self.net.load_state_dict(state["model"] if "model" in state else state)
                self.trained = True
            else:
                self.trained = False  # untrained network: still valid, just not good
            self.net.eval()
            self.h = None
            self.ok = True
            self._warmup()
        except Exception as e:  # noqa: BLE001 - never fail initialisation
            print(f"[agent] init fallback: {type(e).__name__}: {e}", file=sys.stderr)

    def _warmup(self) -> None:
        torch = self.torch
        with torch.no_grad():
            x = {"rays": torch.zeros(1, 1, 64, 7), "ent": torch.zeros(1, 1, 32, 17), "ent_mask": torch.zeros(1, 1, 32, dtype=torch.bool),
                 "body": torch.zeros(1, 1, 26), "map": torch.zeros(1, 1, 14, 32, 32)}
            self.net(x)

    def reset(self) -> None:
        self.prev = np.zeros(self.n, np.int8)
        self.t = 0
        self.h = None
        self.rng = np.random.default_rng(0)
        if self.ok:
            try:
                self.mapper.reset()
            except Exception:  # noqa: BLE001
                pass

    def probs(self, observation: dict) -> np.ndarray:
        """Button probabilities for this step (also advances the memory)."""
        torch = self.torch
        self.mapper.update(observation, self.prev if self.t > 0 else None)
        x = self.encode(observation, self.mapper.crops(), self.prev)
        self.last_x = x  # training reads the encoded observation from here
        with torch.no_grad():
            xt = {k: torch.from_numpy(np.ascontiguousarray(v))[None, None] for k, v in x.items()}
            logits, _, self.h = self.net(xt, self.h)
        p = torch.sigmoid(logits[0, 0]).numpy().astype(np.float64)
        return np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)

    def mask(self, a: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Action masking from the agent's own state (allowed by docs/RULES.md)."""
        if self.prev[USE]:
            a[USE] = 0  # USE only fires on the button-down edge; holding it does nothing
        if self.prev[NEXTW]:
            a[NEXTW] = 0
        for i, j in ((FWD, BACK), (LEFT, RIGHT), (TL, TR)):
            if a[i] and a[j]:  # opposite buttons cancel: keep the one the network prefers
                a[j if p[i] >= p[j] else i] = 0
        return a

    def act(self, observation: dict) -> np.ndarray:
        a = np.zeros(self.n, np.int8)
        try:
            if self.ok:
                p = self.probs(observation)
                a = self.mask((self.rng.random(self.n) < p).astype(np.int8), p)
        except Exception as e:  # noqa: BLE001 - a fault must not end the map
            if self.t % 100 == 0:
                print(f"[agent] act fallback: {type(e).__name__}: {e}", file=sys.stderr)
            a = np.zeros(self.n, np.int8)
            a[FWD] = 1
        self.prev = a
        self.t += 1
        return a
