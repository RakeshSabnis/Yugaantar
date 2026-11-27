"""Scripted teacher (TRAINING ONLY, never used by the submitted agent).

It reads privileged ground truth (the map grid and the player's true position and heading) and plays
the map: explore towards unseen space, pick up needed keys, open doors, reach the exit, and fight
visible enemies. The network learns to copy its decisions from the legal observations alone.

To keep the copy learnable, the teacher only acts on what it has *seen* so far (a line-of-sight
"seen" mask over the grid), not on the full map: it doesn't walk straight to a key it couldn't know about.
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submission.actions import COMBOS, F, TL, TR, SL, SR, ATK, USE, NEXTW, B, from_buttons  # noqa: E402
from mapgen import CELL, KEY_FOR_LOCK, LOCKS  # noqa: E402

KEY_VITAL = {"r": 10, "b": 8, "y": 9}  # vitals index of the card flag (red, blue, yellow)
ENEMY_TYPES = (1, 2, 3)
FIRE_WHILE_ALIGNING = __import__("os").environ.get("TEACHER_FIRE_ALIGN", "0") == "1"


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class Teacher:
    def __init__(self, grid_text: str, rng: np.random.Generator | None = None):
        rows = grid_text.strip("\n").split("\n")
        self.h, self.w = len(rows), max(len(r) for r in rows)
        self.grid = [list(r.ljust(self.w, "#")) for r in rows]
        self.rng = rng or np.random.default_rng(0)
        self.seen = np.zeros((self.h, self.w), bool)
        self.visited = np.zeros((self.h, self.w), bool)
        self.prev_use = False
        self.prev_action = 0
        self.stuck_steps = 0
        self.unstick = 0
        self.unstick_action = 0
        self.last_pos = None
        self.strafe_dir = 1
        self.strafe_timer = 0
        self.door_wait = 0
        self.goal = None
        self.goal_kind = ""
        self.dir_rel = None
        self.t = 0
        self.blocked: dict[tuple[int, int], int] = {}  # door cells that would not open -> retry after step

    # ------------------------------------------------------------------ geometry helpers
    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return self.h - 1 - int(y // CELL), int(x // CELL)

    def center(self, r: int, c: int) -> tuple[float, float]:
        return c * CELL + CELL / 2, (self.h - 1 - r) * CELL + CELL / 2

    def at(self, r, c):
        return self.grid[r][c] if 0 <= r < self.h and 0 <= c < self.w else "#"

    def _update_seen(self, x: float, y: float, door_open) -> None:
        for k in range(48):
            a = 2 * math.pi * k / 48
            dx, dy = math.cos(a), math.sin(a)
            for s in range(1, 64):
                px, py = x + dx * s * 16, y + dy * s * 16
                r, c = self.cell_of(px, py)
                if not (0 <= r < self.h and 0 <= c < self.w):
                    break
                self.seen[r, c] = True
                ch = self.grid[r][c]
                if ch == "#" or (ch in "DRBY" and not door_open(r, c)):
                    break

    def _passable(self, r, c, keys: set[str]) -> bool:
        ch = self.at(r, c)
        if ch == "#":
            return False
        if self.blocked.get((r, c), -1) > self.t:
            return False
        if ch in LOCKS and KEY_FOR_LOCK[ch] not in keys:
            return False
        return True

    def _bfs(self, start, keys, targets_fn, known_only=True):
        """Shortest 4-connected path over seen, passable cells to the first cell where targets_fn is true."""
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur != start and targets_fn(cur):
                path = [cur]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            r, c = cur
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if n in prev or not self._passable(*n, keys):
                    continue
                if known_only and not self.seen[n]:
                    continue
                prev[n] = cur
                q.append(n)
        return None

    # ------------------------------------------------------------------ main decision
    def act(self, obs: dict, priv: dict, door_open) -> int:
        self.t += 1
        x, y, ang = priv["x"], priv["y"], priv["angle"]
        cell = self.cell_of(x, y)
        self._update_seen(x, y, door_open)
        if 0 <= cell[0] < self.h and 0 <= cell[1] < self.w:
            self.visited[cell] = True
        vit = obs["vitals"]
        keys = {k for k, i in KEY_VITAL.items() if vit[i] > 0.5 or vit[i + 3] > 0.5}

        # stuck detection
        if self.last_pos is not None and math.hypot(x - self.last_pos[0], y - self.last_pos[1]) < 1.0 and COMBOS[self.prev_action] and F in COMBOS[self.prev_action]:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        self.last_pos = (x, y)

        self.dir_rel = None  # ego-centric direction (degrees, + = left) of the current waypoint, if navigating
        action = self._decide(obs, x, y, ang, cell, keys, door_open)
        if USE in COMBOS[action] and self.prev_use:
            action = from_buttons([1 if i == F else 0 for i in range(9)]) if F in COMBOS[action] else 0
        self.prev_use = USE in COMBOS[action]
        self.prev_action = action
        return action

    def _decide(self, obs, x, y, ang, cell, keys, door_open) -> int:
        if self.unstick > 0:
            self.unstick -= 1
            return self.unstick_action
        if self.stuck_steps > 8:
            self.stuck_steps = 0
            self.unstick = 4
            self.unstick_action = int(self.rng.choice([6, 7, 8, 20, 21]))
            return self.unstick_action

        # Out of ammo for the current weapon: cycle weapons (edge-triggered).
        vit = obs["vitals"]
        if vit[3] <= 0 and (vit[4] > 0 or vit[5] > 0) and NEXTW not in COMBOS[self.prev_action]:
            return 19

        fight = self._combat(obs)
        if fight is not None:
            return fight

        path = self._plan(cell, keys)
        if path is None or len(path) < 2:
            return int(self.rng.choice([4, 5, 1]))
        return self._follow(path, x, y, ang, keys, door_open, obs)

    def _dodge(self, obs) -> int | None:
        """Incoming projectile (imp fireball etc.) on a collision course: sidestep it, keep shooting."""
        ents = obs["visible_entities"]
        for k in range(len(ents["mask"])):
            if not ents["mask"][k] or ents["entity_type"][k] != 4:
                continue
            d, s, c, vf, vr = (float(v) for v in ents["features"][k])
            dist = d * 1024
            if dist > 600:
                continue
            px, py = c * dist, s * dist          # position: forward, left
            vx, vy = vf, -vr                     # relative velocity: forward, left
            closing = -(px * vx + py * vy) / max(dist, 1.0)
            if closing < 2.0:
                continue
            # closest approach of its straight-line path to us
            miss = abs(px * vy - py * vx) / max(math.hypot(vx, vy), 1e-6)
            if miss > 48:
                continue
            # step to the side away from the projectile's path; strafe left = +y
            side_left = (px * vy - py * vx) < 0
            return 14 if side_left else 15  # attack + strafe
        return None

    def _combat(self, obs) -> int | None:
        dodge = self._dodge(obs)
        if dodge is not None:
            return dodge
        ents = obs["visible_entities"]
        et = np.asarray(ents["entity_type"])
        feat = np.asarray(ents["features"])
        enemy = (np.asarray(ents["mask"]) > 0) & (et >= 1) & (et <= 3)
        has_ammo = obs["vitals"][3] > 0
        if not enemy.any() or not has_ammo:
            # enemy seen by rays but outside the field of view: turn toward it if close
            hits = np.flatnonzero((obs["rays"]["hit_type"] == 4) & (obs["rays"]["distance"] < 0.45))
            if len(hits) and has_ammo:
                k = int(hits[np.argmin(obs["rays"]["distance"][hits])])
                return 4 if k < 32 else 5
            return None
        idx = np.flatnonzero(enemy)  # sorted nearest first
        near = [k for k in idx if feat[k, 0] * 1024.0 < 160.0]
        hitscan = [k for k in idx if et[k] == 2]
        i = int((near or hitscan or list(idx))[0])  # point-blank threat, else hitscanners, else nearest
        dist = float(feat[i, 0]) * 1024.0
        rel = math.degrees(math.atan2(float(feat[i, 1]), float(feat[i, 2])))
        if not (dist < 900.0 or (dist < 1400.0 and et[i] != 1)):
            return None
        tol = math.degrees(math.atan2(18.0, max(dist, 1.0))) + 1.5
        if abs(rel) > 7.5:  # coarse: the smallest turn is ~7 degrees
            return 4 if rel > 0 else 5
        self.strafe_timer -= 1
        if self.strafe_timer <= 0:
            self.strafe_dir = -self.strafe_dir
            self.strafe_timer = int(self.rng.integers(8, 16))
        if et[i] == 1 and dist < 130:
            return 22  # melee monster at arm's length: back off while shooting
        if abs(rel) > max(tol, 3.0):
            if FIRE_WHILE_ALIGNING and dist < 400:
                return 14 if rel > 0 else 15  # close enough that a slightly-off shot still lands
            return 7 if rel > 0 else 8  # fine aim by side-stepping (no shot wasted)
        if abs(rel) > tol:
            return 14 if rel > 0 else 15  # nearly on target: shoot while side-stepping into line
        if dist > 150 and et[i] != 1:
            return 14 if self.strafe_dir > 0 else 15  # on target: shoot and dodge
        return 11

    def _plan(self, cell, keys):
        g = self.at
        # 1. exit if known and reachable
        exits = lambda p: g(*p) == "E"  # noqa: E731
        path = self._bfs(cell, keys, exits)
        if path:
            self.goal_kind = "exit"
            return path
        # 2. needed keys that have been seen and not picked up
        needed = {KEY_FOR_LOCK[ch] for r in range(self.h) for ch in self.grid[r] if ch in LOCKS} - keys
        if needed:
            path = self._bfs(cell, keys, lambda p: g(*p) in needed)
            if path:
                self.goal_kind = "key"
                return path
        # 3. frontier: seen passable cell next to an unseen cell
        def frontier(p):
            r, c = p
            return any(not self.seen[r + dr, c + dc] and self.at(r + dr, c + dc) != "#"
                       for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                       if 0 <= r + dr < self.h and 0 <= c + dc < self.w)
        path = self._bfs(cell, keys, frontier)
        if path:
            self.goal_kind = "frontier"
            return path
        # 4. everything seen: walk to the least-visited reachable cell
        path = self._bfs(cell, keys, lambda p: not self.visited[p])
        self.goal_kind = "revisit"
        return path

    def _follow(self, path, x, y, ang, keys, door_open, obs) -> int:
        # Look ahead along the path while the straight line stays in open cells.
        target = path[1]
        for p in path[2:5]:
            if self._clear_line(path[0], p):
                target = p
            else:
                break
        # Next cell is a closed door: stand in this cell, face the door squarely, press USE (edge),
        # then wait for it to open; re-press only if it is still shut after a while.
        p = path[1]
        if self.at(*p) in "DRBY" and not door_open(*p):
            tx, ty = self.center(*p)
            dist = math.hypot(tx - x, ty - y)
            rel = self.dir_rel = _wrap(math.degrees(math.atan2(ty - y, tx - x)) - ang)
            if abs(rel) > 12:
                return 4 if rel > 0 else 5
            if dist > 70:
                return 1
            self.door_wait += 1
            if self.door_wait > 70:  # won't open (e.g. missing key): avoid it for a while
                self.blocked[p] = self.t + 300
                self.door_wait = 0
                return 0
            if self.door_wait % 20 == 1 and not self.prev_use:
                return 17
            return 0
        if self.at(*p) not in "DRBY":
            self.door_wait = 0
        # Lookahead must not skip over a door cell that is still closed.
        if any(self.at(*q) in "DRBY" and not door_open(*q) for q in path[1:path.index(target) + 1]):
            target = path[1]
        tx, ty = self.center(*target)
        rel = self.dir_rel = _wrap(math.degrees(math.atan2(ty - y, tx - x)) - ang)
        if rel > 25:
            return 4 if abs(rel) > 60 else 2
        if rel < -25:
            return 5 if abs(rel) > 60 else 3
        if rel > 8:
            return 2
        if rel < -8:
            return 3
        return 1

    def _clear_line(self, a, b) -> bool:
        (r0, c0), (r1, c1) = a, b
        n = max(abs(r1 - r0), abs(c1 - c0)) * 3
        for k in range(n + 1):
            t = k / max(n, 1)
            r, c = round(r0 + (r1 - r0) * t), round(c0 + (c1 - c0) * t)
            for dr in (-0.3, 0.3):
                for dc in (-0.3, 0.3):
                    if self.at(int(round(r0 + (r1 - r0) * t + dr)), int(round(c0 + (c1 - c0) * t + dc))) in "#DRBY":
                        return False
            if self.at(r, c) in "#DRBY":
                return False
        return True


def dir_bin(rel) -> int:
    """Direction label: 0..15 = 22.5-degree bins (bin 0 = straight ahead, counter-clockwise), 16 = none."""
    if rel is None:
        return 16
    return int(((rel % 360.0) + 11.25) // 22.5) % 16


def door_open_fn(env, grid_text: str):
    """Returns door_open(r, c): is the door sector at grid cell (r, c) open (ceiling raised)?"""
    import sys as _s

    _s.path.insert(0, str(Path(__file__).parent))
    rows = grid_text.strip("\n").split("\n")
    h = len(rows)
    geo = env._geometry
    p1 = geo.p1
    # door cell -> its sector: find a door linedef whose midpoint lies on this cell's boundary.
    cache: dict = {}

    def sector_at(r, c):
        if (r, c) in cache:
            return cache[(r, c)]
        x0, y0 = c * CELL, (h - 1 - r) * CELL
        mids = (geo.p1 + geo.p2) / 2
        inside = (mids[:, 0] >= x0 - 0.5) & (mids[:, 0] <= x0 + CELL + 0.5) & (mids[:, 1] >= y0 - 0.5) & (mids[:, 1] <= y0 + CELL + 0.5)
        sid = None
        for i in np.flatnonzero(inside):
            for s in (geo.sec_a[i], geo.sec_b[i]):
                if s >= 0 and geo.door_lock[s] >= 0:
                    sid = int(s)
        cache[(r, c)] = sid
        return sid

    def door_open(r, c):
        s = sector_at(r, c)
        if s is None:
            return True
        return bool(geo.ceil[s] - geo.floor[s] >= 56)

    return door_open
