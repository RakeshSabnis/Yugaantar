"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Every decision comes from the trained network (submission/model.py): no scripted rules, pathfinding or
aiming. The only non-learned logic is safety: input sanitising, an exception guard that always returns a
valid action, and action masking from the agent's own previous action (USE / weapon switch are
edge-triggered, so pressing them twice in a row is masked out). Sampling uses one fixed temperature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
sys.path.insert(0, str(SUBMISSION_DIR))

import torch  # noqa: E402

from actions import NEXTW_ACTIONS, NUM_ACTIONS, USE_ACTIONS, to_buttons  # noqa: E402
from model import Perception, Policy, load_compatible  # noqa: E402

TEMPERATURE = float(__import__("os").environ.get("DOOM_TEMPERATURE", "0.5"))  # low-temperature sampling: mostly the best action, enough noise to escape loops


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str, weights_path: str | None = None):
        """Called once. Inference always runs on CPU (one thread): the model is small, and this avoids
        an untested CUDA path and any VRAM limit."""
        self.observation_space = observation_space
        self.action_space = action_space
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        # One or more trained networks. With several, their action probabilities are averaged (an ensemble):
        # every decision still comes from trained parameters, and averaging makes play more consistent.
        if weights_path:
            paths = [Path(p) for p in str(weights_path).split(",")]
        else:
            paths = sorted(WEIGHTS_DIR.glob("policy*.pt"))
        self.models = []
        self.loaded = False
        for path in paths:
            if path.is_file():
                m = Policy()
                state = torch.load(path, map_location="cpu", weights_only=True)
                load_compatible(m, state["model"] if "model" in state else state)
                m.eval()
                self.models.append(m)
                self.loaded = True
        if not self.models:
            m = Policy()
            m.eval()
            self.models.append(m)
        self.model = self.models[0]
        self.rng = np.random.default_rng(0)
        self.perception = Perception()
        self.reset()
        # Warm-up so the first act() of a map is fast.
        dummy = {
            "rays": {"distance": np.ones(64, np.float32), "hit_type": np.zeros(64, np.int64)},
            "visible_entities": {"features": np.zeros((32, 5), np.float32), "entity_type": np.zeros(32, np.int64), "mask": np.zeros(32, np.int8)},
            "vitals": np.zeros(17, np.float32),
        }
        for _ in range(3):
            self.act(dummy)
        self.reset()

    def reset(self) -> None:
        """Called before every map: clear the recurrent memory and the action history."""
        self.hs = [m.initial_state(1) for m in self.models]
        self.prev_action = 0
        self.rng = np.random.default_rng(0)
        self.perception.reset()

    def act(self, observation: dict) -> np.ndarray:
        try:
            x = torch.from_numpy(self.perception.see(observation)).unsqueeze(0)
            pa = torch.tensor([self.prev_action])
            probs = None
            for k, m in enumerate(self.models):
                lg, _, self.hs[k] = m.step(x, pa, self.hs[k])
                p_k = torch.softmax(lg[0].double(), -1).numpy()
                probs = p_k if probs is None else probs + p_k
            logits = np.log(np.maximum(probs / len(self.models), 1e-12))  # log of the averaged probabilities
            if self.prev_action in USE_ACTIONS:
                logits[list(USE_ACTIONS)] = -np.inf
            if self.prev_action in NEXTW_ACTIONS:
                logits[list(NEXTW_ACTIONS)] = -np.inf
            if not np.all(np.isfinite(logits[np.isfinite(logits)])) or not np.isfinite(logits).any():
                raise ValueError("bad logits")
            z = (logits - logits[np.isfinite(logits)].max()) / TEMPERATURE
            p = np.exp(z)
            p /= p.sum()
            a = int(self.rng.choice(NUM_ACTIONS, p=p))
            self.prev_action = a
            buttons = to_buttons(a)
            self.perception.did(buttons)
            return buttons
        except Exception:  # never crash: a crash freezes the player for the rest of the map
            self.prev_action = 1
            self.hs = [m.initial_state(1) for m in self.models]
            self.perception.did(to_buttons(1))
            return to_buttons(1)  # move forward
