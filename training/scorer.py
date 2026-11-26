"""Local scorer: an estimate of the organisers' scoring, computed from privileged ground truth.

Categories follow docs/EVALUATION.md (exit, progression, exploration, combat & survival). The exact
weights are hidden, so these are our own estimates. They are used only to compare checkpoints with
each other on the same held-out maps, never as a claim about the real leaderboard.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

TIER_WEIGHT = {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0}
AREA = 128.0  # exploration grid size in map units


@dataclass
class EpisodeResult:
    map: str
    tier: int
    exit: bool
    dead: bool
    keys: int
    doors_opened: int
    locks_opened: int
    areas: int
    kills: int
    damage: float
    health: float
    tic: int
    ticks: int
    crashed: bool = False

    def score(self) -> float:
        s = 0.0
        if self.exit:
            s += 100.0 + 10.0 * min(self.health, 100) / 100.0 + 10.0 * (1.0 - self.tic / self.ticks)
        s += 15.0 * self.keys + 5.0 * self.doors_opened + 15.0 * self.locks_opened
        s += min(30.0, 1.0 * self.areas)
        s += 4.0 * self.kills
        s -= 0.05 * self.damage
        if self.dead:
            s -= 25.0 * (1.0 - self.tic / self.ticks)
        return s

    def as_dict(self) -> dict:
        d = asdict(self)
        d["score"] = self.score()
        return d


class Tracker:
    """Feed it privileged_state() every step; it accumulates the metrics."""

    def __init__(self, priv0: dict):
        self.door_lock = priv0["door_lock"]
        self.opened = np.zeros(len(self.door_lock), bool)
        self.areas: set = set()
        self.update(priv0)

    def update(self, p: dict) -> None:
        self.areas.add((math.floor(p["x"] / AREA), math.floor(p["y"] / AREA)))
        if p["sector_ceiling"] is not None:
            is_door = self.door_lock >= 0
            self.opened |= is_door & (p["sector_ceiling"] - p["sector_floor"] >= 56)
        self.last = p

    def result(self, map_name: str, tier: int, crashed: bool = False) -> EpisodeResult:
        p = self.last
        return EpisodeResult(
            map=map_name, tier=tier, exit=bool(p["exit_reached"]), dead=bool(p["dead"]),
            keys=int(np.sum(p["keys"])), doors_opened=int(np.sum(self.opened & (self.door_lock == 0))),
            locks_opened=int(np.sum(self.opened & (self.door_lock > 0))), areas=len(self.areas),
            kills=int(p["kills"]), damage=float(p["damage_taken"]), health=float(p["health"]),
            tic=int(p["tic"]), ticks=int(p["episode_ticks"]), crashed=crashed,
        )


def summarize(results: list[EpisodeResult]) -> dict:
    out = {"maps": len(results), "weighted": 0.0}
    for tier in sorted({r.tier for r in results}):
        rs = [r for r in results if r.tier == tier]
        out[f"t{tier}"] = {
            "score": float(np.mean([r.score() for r in rs])),
            "exit": float(np.mean([r.exit for r in rs])),
            "keys": float(np.mean([r.keys for r in rs])),
            "kills": float(np.mean([r.kills for r in rs])),
            "dead": float(np.mean([r.dead for r in rs])),
            "areas": float(np.mean([r.areas for r in rs])),
            "crashed": int(sum(r.crashed for r in rs)),
        }
        out["weighted"] += TIER_WEIGHT[tier] * out[f"t{tier}"]["score"]
    return out
