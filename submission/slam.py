"""Lidar-style mapping from the official observations (no privileged data).

The observations contain no position or heading, so we estimate them ourselves (odometry):
- heading from our OWN turn actions: in Doom, holding a turn button turns 7.03° on the first step,
  12.30° on the second and 14.06° per step after that (measured; switching direction keeps the rate);
- position by integrating the observed ego-frame velocity (vitals vel_forward / vel_right).

Each step the 64 rays are painted into an occupancy grid in that odometry frame: cells a ray passes
through become "free", the hit cell gets the hit type (wall, door, locked door). We also keep a trail of
where we have been. The network reads an ego-centric crop of this map (rotated to our heading), so it
can remember where the locked door was and which areas are still unexplored.

This is observation preprocessing / memory. It never chooses actions: the network does.
"""

from __future__ import annotations

import math

import numpy as np

TURN_DEG = (7.03125, 12.3046875, 14.0625)  # per step for the 1st, 2nd, 3rd+ consecutive turning step
RAY_RANGE = 1024.0
NUM_RAYS = 64
CELL = 64.0           # map resolution (map units per cell; matches Doom's 64-unit grid)
SIZE = 256            # world grid is SIZE x SIZE cells (16384 units), origin at the centre
CROP = 32             # ego-centric crop is CROP x CROP cells...
CROP_CELL = 64.0      # ...each covering 64 map units (so the crop spans 2048 x 2048 units)
CHANNELS = 6          # free, wall, door, locked door, visited, unknown
WIDE = 16             # zoomed-out crop: WIDE x WIDE coarse cells...
WIDE_CELL = 256.0     # ...each 256 map units (4 x 4 fine cells), spanning 4096 x 4096 units around us

F_IDX, B_IDX, SL_IDX, SR_IDX, TL_IDX, TR_IDX = 0, 1, 2, 3, 4, 5
USE_IDX = 7
_RAY_ANG = np.arange(NUM_RAYS) * (2 * math.pi / NUM_RAYS)
_SAMPLES = np.linspace(0.0, 1.0, 40, endpoint=False)[1:]  # fractions along each ray for free-space painting


class Slam:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.free = np.zeros((SIZE, SIZE), np.uint8)
        self.hit = np.zeros((SIZE, SIZE), np.uint8)   # 0 none, 1 wall, 2 door, 3 locked door
        self.visit = np.zeros((SIZE, SIZE), np.uint8)
        self.tried = np.zeros((SIZE, SIZE), np.uint8)  # locked doors we pressed USE on that stayed shut
        self.pending: list = []   # (cell x, cell y, step at which to check whether the door opened)
        self.used_last = False    # did our last action press USE?
        self.keys_n = 0           # how many key flags are set in the vitals
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0  # radians, odometry frame (0 = facing direction at the start of the map)
        self.turn_streak = 0
        self.steps = 0
        self.prev_d = None
        self.pred_turn = 0.0  # heading change predicted from our action (radians)
        self.pvx = self.pvy = 0.0

    # ------------------------------------------------------------------ odometry
    def apply_action(self, buttons) -> None:
        """Call with the buttons we just sent, before the next observation arrives."""
        b = np.asarray(buttons).reshape(-1)
        self.used_last = bool(b[USE_IDX] > 0)
        turn = int(b[TL_IDX] > 0) - int(b[TR_IDX] > 0)
        if turn != 0:
            self.turn_streak += 1
            deg = TURN_DEG[min(self.turn_streak, 3) - 1]
            if self.steps <= 1:  # the engine applies little turning in the first steps of a map
                deg = 0.0 if self.steps == 0 else 3.515625
            self.pred_turn = math.radians(deg) * turn
        else:
            self.pred_turn = 0.0
            self.turn_streak = 0
        self.steps += 1

    def _scan_match(self, d: np.ndarray, valid: np.ndarray) -> float:
        """Refine the predicted turn by rotating the previous scan to best match the current one."""
        pred = self.pred_turn
        if self.prev_d is None:
            return pred
        prev, pvalid = self.prev_d
        step = 2 * math.pi / NUM_RAYS
        best, best_err = pred, None
        cands = pred + np.radians(np.arange(-9.0, 9.01, 0.5))
        idx = np.arange(NUM_RAYS)
        for delta in cands:
            pos = (idx + delta / step) % NUM_RAYS  # new ray at angle a saw what the old ray at a + delta saw
            i0 = np.floor(pos).astype(np.int64) % NUM_RAYS
            i1 = (i0 + 1) % NUM_RAYS
            f = pos - np.floor(pos)
            old = prev[i0] * (1 - f) + prev[i1] * f
            m = valid & pvalid[i0] & pvalid[i1]
            if m.sum() < 16:
                return pred
            err = float(np.median(np.abs(old[m] - d[m]) / (d[m] + 64.0)))
            if best_err is None or err < best_err:
                best, best_err = delta, err
        pred_err = None
        # keep the action prediction unless the scan clearly prefers another angle
        pos = (idx + pred / step) % NUM_RAYS
        i0 = np.floor(pos).astype(np.int64) % NUM_RAYS
        i1 = (i0 + 1) % NUM_RAYS
        f = pos - np.floor(pos)
        old = prev[i0] * (1 - f) + prev[i1] * f
        m = valid & pvalid[i0] & pvalid[i1]
        pred_err = float(np.median(np.abs(old[m] - d[m]) / (d[m] + 64.0)))
        return best if best_err < 0.8 * pred_err else pred

    def observe(self, obs: dict) -> None:
        """Correct the heading by scan matching, integrate velocity, then paint the rays."""
        d = np.clip(np.nan_to_num(np.asarray(obs["rays"]["distance"], np.float64).reshape(NUM_RAYS), nan=1.0), 0.0, 1.0) * RAY_RANGE
        ht = np.asarray(obs["rays"]["hit_type"], np.int64).reshape(NUM_RAYS)
        valid = (d < RAY_RANGE - 1) & (ht >= 1) & (ht <= 3)  # static geometry only (not monsters/items)
        self.heading += self.pred_turn  # action model (scan matching: see _scan_match, not reliable enough)
        self.pred_turn = 0.0
        self.prev_d = (d, valid)
        v = np.asarray(obs["vitals"], np.float64).reshape(-1)
        fwd, right = float(np.nan_to_num(v[14])), float(np.nan_to_num(v[15]))
        fwd, right = max(-40.0, min(40.0, fwd)), max(-40.0, min(40.0, right))
        c, s = math.cos(self.heading), math.sin(self.heading)
        vx, vy = fwd * c + right * s, fwd * s - right * c
        # Displacement over the 4 tics of a step, fitted against true motion (Doom friction makes the
        # end-of-step velocity an incomplete summary): 2.589 * v_now + 1.801 * v_prev.
        self.x += 2.589 * vx + 1.801 * self.pvx
        self.y += 2.589 * vy + 1.801 * self.pvy
        self.pvx, self.pvy = vx, vy
        self._paint(obs)

    def _cell(self, x, y):
        # +CELL/2: the start position is the centre of a map cell, so walls fall on cell borders
        return (np.floor((np.asarray(x) + CELL / 2) / CELL).astype(np.int64) + SIZE // 2,
                np.floor((np.asarray(y) + CELL / 2) / CELL).astype(np.int64) + SIZE // 2)

    def _paint(self, obs: dict) -> None:
        rays = obs["rays"]
        d = np.clip(np.nan_to_num(np.asarray(rays["distance"], np.float64).reshape(NUM_RAYS), nan=1.0), 0.0, 1.0) * RAY_RANGE
        ht = np.asarray(rays["hit_type"], np.int64).reshape(NUM_RAYS)
        ang = self.heading + _RAY_ANG
        dx, dy = np.cos(ang), np.sin(ang)
        # free space along each ray (stop a little short of the hit)
        frac = _SAMPLES[None, :] * np.maximum(d - 8.0, 0.0)[:, None]
        fx = self.x + dx[:, None] * frac
        fy = self.y + dy[:, None] * frac
        cx, cy = self._cell(fx.ravel(), fy.ravel())
        ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
        self.free[cy[ok], cx[ok]] = 1
        # a ray passing through a cell means any door there is open now: drop the stale door mark
        stale = self.hit[cy[ok], cx[ok]] >= 2
        if stale.any():
            self.hit[cy[ok][stale], cx[ok][stale]] = 0
            self.tried[cy[ok][stale], cx[ok][stale]] = 0
        # hits: walls / doors / locked doors (enemies and items are transient: not mapped)
        solid = (ht >= 1) & (ht <= 3) & (d < RAY_RANGE - 1)
        if solid.any():
            hx = self.x + dx[solid] * (d[solid] + 4.0)
            hy = self.y + dy[solid] * (d[solid] + 4.0)
            cx, cy = self._cell(hx, hy)
            ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
            self.hit[cy[ok], cx[ok]] = ht[solid][ok].astype(np.uint8)
        # an open door the ray now passes through: clear stale door marks along free space
        # Memory of failed attempts: we pressed USE facing a locked door; if it is still marked locked a
        # little later (a door that accepts our key is open by then), remember it as "tried".
        keys_n = int(np.sum(np.nan_to_num(np.asarray(obs["vitals"], np.float64).reshape(-1)[8:14]) > 0.5))
        if keys_n > self.keys_n:
            self.tried[:] = 0      # a new key: every locked door is worth trying again
            self.pending = []
        self.keys_n = keys_n
        if self.used_last and ht[0] == 3 and d[0] < 110.0:
            tx, ty = self._cell(self.x + dx[0] * (d[0] + 4.0), self.y + dy[0] * (d[0] + 4.0))
            self.pending.append((int(tx), int(ty), self.steps + 12))
        self.used_last = False
        if self.pending:
            keep = []
            for tx, ty, due in self.pending:
                if self.steps < due:
                    keep.append((tx, ty, due))
                    continue
                y0, y1, x0, x1 = max(0, ty - 1), min(SIZE, ty + 2), max(0, tx - 1), min(SIZE, tx + 2)
                self.tried[y0:y1, x0:x1] |= (self.hit[y0:y1, x0:x1] == 3).astype(np.uint8)
            self.pending = keep
        px, py = self._cell(self.x, self.y)
        if 0 <= px < SIZE and 0 <= py < SIZE:
            self.visit[max(0, py - 1):py + 2, max(0, px - 1):px + 2] = 1

    # ------------------------------------------------------------------ network input
    def crop_packed(self) -> np.ndarray:
        """(CROP, CROP) uint8 ego-centric map, one bit per layer: 1 free, 2 wall, 4 door, 8 locked door,
        16 visited, 32 frontier (seen floor next to unexplored space), 64 we hold a key (whole plane),
        128 locked door we tried and it stayed shut. Unknown = no free/hit bit. Row 0 = ahead, column 0 = left."""
        k = (np.arange(CROP) - CROP / 2 + 0.5) * CROP_CELL
        fwd = -k[:, None] * np.ones((1, CROP))
        left = -k[None, :] * np.ones((CROP, 1))
        c, s = math.cos(self.heading), math.sin(self.heading)
        cx, cy = self._cell(self.x + fwd * c - left * s, self.y + fwd * s + left * c)
        ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
        cx, cy = np.clip(cx, 0, SIZE - 1), np.clip(cy, 0, SIZE - 1)
        hit = self.hit[cy, cx]
        free = self.free[cy, cx]
        # frontier: seen floor that touches a never-seen cell = where unexplored space begins
        known = (self.free > 0) | (self.hit > 0)
        unknown_nb = np.zeros_like(free, dtype=bool)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            unknown_nb |= ~known[np.clip(cy + dy, 0, SIZE - 1), np.clip(cx + dx, 0, SIZE - 1)]
        frontier = (free > 0) & (hit == 0) & unknown_nb
        out = (free | ((hit == 1) << 1) | ((hit == 2) << 2) | ((hit == 3) << 3) | (self.visit[cy, cx] << 4) | (frontier.astype(np.uint8) << 5)
               | (int(self.keys_n > 0) << 6) | (self.tried[cy, cx].astype(np.int64) << 7))
        return (out * ok).astype(np.uint8)

    def wide_packed(self) -> np.ndarray:
        """(WIDE, WIDE) uint8 zoomed-out ego-centric view, one bit per layer, each true if ANY of the
        4x4 fine cells under the coarse cell has it: 1 seen floor, 2 visited, 4 frontier, 8 door,
        16 locked door, 32 wall, 64 we hold a key (whole plane), 128 a tried locked door. Lets the network see unexplored space well beyond the fine crop."""
        sub = (np.arange(4) - 1.5) * CELL
        k = (np.arange(WIDE) - WIDE / 2 + 0.5) * WIDE_CELL
        fwd = (-k[:, None, None, None] + sub[None, None, :, None]) * np.ones((WIDE, WIDE, 4, 4))
        left = (-k[None, :, None, None] + sub[None, None, None, :]) * np.ones((WIDE, WIDE, 4, 4))
        c, s = math.cos(self.heading), math.sin(self.heading)
        cx, cy = self._cell(self.x + fwd * c - left * s, self.y + fwd * s + left * c)
        ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
        cx, cy = np.clip(cx, 0, SIZE - 1), np.clip(cy, 0, SIZE - 1)
        free = (self.free[cy, cx] > 0) & ok
        hit = self.hit[cy, cx] * ok
        known = (self.free > 0) | (self.hit > 0)
        unknown_nb = np.zeros_like(free)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            unknown_nb |= ~known[np.clip(cy + dy, 0, SIZE - 1), np.clip(cx + dx, 0, SIZE - 1)]
        frontier = free & (hit == 0) & unknown_nb
        anyf = lambda a: a.any(axis=(2, 3)).astype(np.uint8)  # noqa: E731
        return (anyf(free) | (anyf((self.visit[cy, cx] > 0) & ok) << 1) | (anyf(frontier) << 2)
                | (anyf(hit == 2) << 3) | (anyf(hit == 3) << 4) | (anyf(hit == 1) << 5)
                | (np.uint8(self.keys_n > 0) << 6) | (anyf((self.tried[cy, cx] > 0) & ok) << 7)).astype(np.uint8)

    def crop(self) -> np.ndarray:
        """(CHANNELS, CROP, CROP) float32 ego-centric map: row 0 = ahead, column 0 = left."""
        k = (np.arange(CROP) - CROP / 2 + 0.5) * CROP_CELL
        fwd = -k[:, None] * np.ones((1, CROP))          # rows: ahead (top) to behind (bottom)
        left = -k[None, :] * np.ones((CROP, 1))         # cols: left to right
        c, s = math.cos(self.heading), math.sin(self.heading)
        wx = self.x + fwd * c - left * s
        wy = self.y + fwd * s + left * c
        cx, cy = self._cell(wx, wy)
        ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
        cx, cy = np.clip(cx, 0, SIZE - 1), np.clip(cy, 0, SIZE - 1)
        out = np.zeros((CHANNELS, CROP, CROP), np.float32)
        free = self.free[cy, cx] * ok
        hit = self.hit[cy, cx] * ok
        out[0] = free
        out[1] = hit == 1
        out[2] = hit == 2
        out[3] = hit == 3
        out[4] = self.visit[cy, cx] * ok
        out[5] = (free == 0) & (hit == 0)
        return out
