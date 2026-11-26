"""Local stand-in for the hidden scorer.

The real weights are not published; docs/EVALUATION.md only gives the ordering: exit (largest, plus a
small health/time bonus) > progression (keys, doors, locked doors) > exploration (unique areas) >
combat (kills, minus small penalties for damage and dying). The numbers below encode that ordering
so checkpoints can be compared on held-out maps. They are our guess, not the organizers' values.
"""

from __future__ import annotations

import numpy as np

W_EXIT, W_EXIT_BONUS = 100.0, 20.0
W_KEY, W_DOOR, W_LOCKED = 15.0, 4.0, 15.0
W_EXPLORE = 30.0
W_KILLS = 20.0
W_DAMAGE, W_DEATH = 0.05, 10.0
TIER_WEIGHT = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
AREA = 128.0  # exploration cell size, map units


class EpisodeScore:
    """Feed it env.privileged_state() after reset and after every step."""

    def __init__(self, total_monsters: int | None = None, total_area_cells: int | None = None):
        self.total_monsters = total_monsters
        self.total_area = total_area_cells
        self.cells: set = set()
        self.opened: set = set()
        self.state: dict | None = None

    def update(self, st: dict) -> None:
        self.state = st
        self.tic = max(getattr(self, "tic", 0), st["tic"])
        self.cells.add((int(st["x"] // AREA), int(st["y"] // AREA)))
        if st["door_lock"] is not None:
            open_now = np.flatnonzero((st["door_lock"] >= 0) & (st["sector_ceiling"] - st["sector_floor"] >= 56))
            self.opened.update(int(i) for i in open_now)

    def result(self) -> dict:
        st = self.state
        locks = st["door_lock"]
        doors = sum(1 for i in self.opened if locks[i] == 0)
        locked = sum(1 for i in self.opened if locks[i] > 0)
        keys = int(np.sum(st["keys"]))
        exit_pts = 0.0
        if st["exit_reached"]:
            time_left = 1.0 - self.tic / max(1, st["episode_ticks"])
            exit_pts = W_EXIT + W_EXIT_BONUS * 0.5 * (min(1.0, max(0.0, st["health"]) / 100.0) + time_left)
        explore = len(self.cells) / self.total_area if self.total_area else min(1.0, len(self.cells) / 60.0)
        kills = st["kills"] / self.total_monsters if self.total_monsters else min(1.0, st["kills"] / 8.0)
        total = (exit_pts + W_KEY * keys + W_DOOR * doors + W_LOCKED * locked + W_EXPLORE * min(1.0, explore)
                 + W_KILLS * min(1.0, kills) - W_DAMAGE * st["damage_taken"] - (W_DEATH if st["dead"] else 0.0))
        return {"score": float(total), "exit": bool(st["exit_reached"]), "dead": bool(st["dead"]), "keys": keys,
                "doors": doors, "locked": locked, "explore": float(min(1.0, explore)), "kills": int(st["kills"]),
                "damage": float(st["damage_taken"]), "tic": int(self.tic)}
