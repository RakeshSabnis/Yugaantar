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
_RAY_ANG = np.arange(NUM_RAYS) * (2 * math.pi / NUM_RAYS)
DEPTH_UNITS = 7.6     # map units per depth-buffer level (measured)
EXIT_CUE = __import__("os").environ.get("DOOM_EXIT", "1") == "1"  # paint seen exit pads into the map as frontier
_PIX_ROWS = np.arange(62, 120, dtype=np.float64)[:, None]
_PIX_LEFT = ((79.5 - np.arange(160)) / 80.0)[None, :]
_FOV_RAYS = tuple(range(0, 8)) + tuple(range(57, 64))  # rays inside the 90 degree field of view
_SAMPLES = np.linspace(0.0, 1.0, 40, endpoint=False)[1:]  # fractions along each ray for free-space painting


class Slam:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.free = np.zeros((SIZE, SIZE), np.uint8)
        self.hit = np.zeros((SIZE, SIZE), np.uint8)   # 0 none, 1 wall, 2 door, 3 locked door
        self.visit = np.zeros((SIZE, SIZE), np.uint8)
        self.color = np.zeros((SIZE, SIZE), np.uint8)  # locked-door colour: 0 unknown, 1 red, 2 blue, 3 yellow
        self.keys = np.zeros(4, bool)                  # keys held, by the same colour index
        self.exit = np.zeros((SIZE, SIZE), np.uint8)   # cells where an exit pad has been seen
        self.exit_false = 0
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
        k = np.nan_to_num(v[8:14]) > 0.5  # blue card, yellow card, red card, blue skull, yellow skull, red skull
        self.keys[1], self.keys[2], self.keys[3] = bool(k[2] or k[5]), bool(k[0] or k[3]), bool(k[1] or k[4])
        self._paint(obs)

    def _cell(self, x, y):
        return (np.floor(np.asarray(x) / CELL).astype(np.int64) + SIZE // 2,
                np.floor(np.asarray(y) / CELL).astype(np.int64) + SIZE // 2)

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
        # hits: walls / doors / locked doors (enemies and items are transient: not mapped)
        solid = (ht >= 1) & (ht <= 3) & (d < RAY_RANGE - 1)
        if solid.any():
            hx = self.x + dx[solid] * (d[solid] + 4.0)
            hy = self.y + dy[solid] * (d[solid] + 4.0)
            cx, cy = self._cell(hx, hy)
            ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
            self.hit[cy[ok], cx[ok]] = ht[solid][ok].astype(np.uint8)
        self._door_colours(obs, d, ht, dx, dy)
        self._exit_pads(obs)
        # an open door the ray now passes through: clear stale door marks along free space
        px, py = self._cell(self.x, self.y)
        if 0 <= px < SIZE and 0 <= py < SIZE:
            self.visit[max(0, py - 1):py + 2, max(0, px - 1):px + 2] = 1

    # ------------------------------------------------------------------ door colours (from the frame)
    def _door_colours(self, obs: dict, d: np.ndarray, ht: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> None:
        """Rays report "locked door" but not which key it needs. The frame does: locked doors are drawn
        with a red, blue or yellow grille. For every locked-door ray inside the field of view we look
        at the matching image column, keep the pixels whose depth agrees with the ray, and store the
        colour in the map. Without a frame (training with buffers off) nothing is stored."""
        rgb, depth = obs.get("rgb"), obs.get("depth")
        if rgb is None or depth is None:
            return
        rgb, depth = np.asarray(rgb), np.asarray(depth)
        if rgb.shape != (120, 160, 3) or depth.shape != (120, 160) or not rgb.any():
            return
        top = rgb[20:58].astype(np.int16)
        r, g, b = top[..., 0], top[..., 1], top[..., 2]
        red = (r > 80) & (g < 40) & (b < 40)
        if red.mean() > 0.30:
            return  # taking damage tints the whole frame red: do not read colours from it
        masks = (red, (b > 90) & (r < 40) & (g < 40), (r > 120) & (g > 70) & (b < 70) & (r > g + 25))
        dep = depth[20:58].astype(np.float64) * DEPTH_UNITS
        for k in _FOV_RAYS:
            if ht[k] != 3 or d[k] >= RAY_RANGE - 1:
                continue
            th = _RAY_ANG[k] if k < NUM_RAYS // 2 else _RAY_ANG[k] - 2 * math.pi
            col = int(round(79.5 - 80.0 * math.tan(th)))
            if not 2 <= col <= 157:
                continue
            fwd = d[k] * math.cos(th)
            sl = slice(col - 2, col + 3)
            on = np.abs(dep[:, sl] - fwd) < 0.25 * fwd + 16.0  # pixels that are on this door, not in front of it
            counts = [int((m[:, sl] & on).sum()) for m in masks]
            best = int(np.argmax(counts))
            if counts[best] < 6 or counts[best] < 2 * sorted(counts)[-2]:
                continue
            cx, cy = self._cell(self.x + dx[k] * (d[k] + 4.0), self.y + dy[k] * (d[k] + 4.0))
            if 0 <= cx < SIZE and 0 <= cy < SIZE:
                self.color[cy, cx] = best + 1

    # ------------------------------------------------------------------ exit pads (from the frame)
    def _exit_pads(self, obs: dict) -> None:
        """Neither the rays nor the entity list show the exit. The frame does: the exit pad is a
        saturated red floor. Each such pixel is placed with the depth buffer and remembered; the map
        then presents those cells to the network as frontier, i.e. as somewhere worth walking to.
        Upright red things (blood, corpses, fireballs) are rejected because all their pixels share one
        depth, while a pad lying on the floor spans a range of depths along an image column."""
        px, py = self._cell(self.x, self.y)
        if 0 <= px < SIZE and 0 <= py < SIZE:
            if self.exit[py, px]:
                self.exit_false += 1  # stood on a "pad" and the level did not end: the cue is wrong here
                if self.exit_false >= 3:
                    self.exit[:] = 0
            self.exit[py, px] = 0
        if not EXIT_CUE or self.exit_false >= 3:
            return
        rgb, depth = obs.get("rgb"), obs.get("depth")
        if rgb is None or depth is None:
            return
        rgb, depth = np.asarray(rgb), np.asarray(depth)
        if rgb.shape != (120, 160, 3) or depth.shape != (120, 160) or not rgb.any():
            return
        lo = rgb[62:]
        r = lo[..., 0].astype(np.int16)
        red = (r > 80) & (lo[..., 1] < r * 0.35) & (lo[..., 2] < r * 0.35)
        if not red.any():
            return
        top = rgb[:40]
        tr = top[..., 0].astype(np.int16)
        if ((tr > 80) & (top[..., 1] < tr * 0.35) & (top[..., 2] < tr * 0.35)).mean() > 0.10 or red.mean() > 0.35:
            return  # damage tint
        fwd = depth[62:].astype(np.float64) * DEPTH_UNITS
        red &= np.abs(fwd * (_PIX_ROWS - 60.0) - 3800.0) < 600.0  # on the floor plane
        span = np.where(red, fwd, -np.inf).max(0) - np.where(red, fwd, np.inf).min(0)
        red &= ((red.sum(0) >= 2) & (span >= 20.0))[None, :]      # lies flat: spans real ground
        if red.sum() < 3:
            return
        f = fwd[red]
        lf = (fwd * _PIX_LEFT)[red]
        c, s = math.cos(self.heading), math.sin(self.heading)
        cx, cy = self._cell(self.x + f * c - lf * s, self.y + f * s + lf * c)
        ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
        self.exit[cy[ok], cx[ok]] = 1

    def effective_hit(self) -> np.ndarray:
        """The hit grid as the network should see it: a locked door whose colour we have read and whose
        key we do not hold is shown as a wall. It turns back into a locked door when the key is picked up."""
        blocked = (self.hit == 3) & (self.color > 0) & ~self.keys[self.color]
        return np.where(blocked, 1, self.hit).astype(np.uint8)

    def ray_types(self, obs: dict) -> np.ndarray:
        """Ray hit types with the same filter applied (see effective_hit)."""
        d = np.clip(np.nan_to_num(np.asarray(obs["rays"]["distance"], np.float64).reshape(NUM_RAYS), nan=1.0), 0.0, 1.0) * RAY_RANGE
        ht = np.asarray(obs["rays"]["hit_type"], np.int64).reshape(NUM_RAYS).copy()
        sel = np.flatnonzero((ht == 3) & (d < RAY_RANGE - 1))
        if len(sel):
            ang = self.heading + _RAY_ANG[sel]
            cx, cy = self._cell(self.x + np.cos(ang) * (d[sel] + 4.0), self.y + np.sin(ang) * (d[sel] + 4.0))
            ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
            col = np.zeros(len(sel), np.int64)
            col[ok] = self.color[cy[ok], cx[ok]]
            ht[sel[(col > 0) & ~self.keys[col]]] = 1
        return ht

    # ------------------------------------------------------------------ network input
    def crop_packed(self) -> np.ndarray:
        """(CROP, CROP) uint8 ego-centric map, one bit per layer: 1 free, 2 wall, 4 door, 8 locked door,
        16 visited, 32 frontier (seen floor next to unexplored space). Unknown = no free/hit bit. Row 0 = ahead, column 0 = left."""
        k = (np.arange(CROP) - CROP / 2 + 0.5) * CROP_CELL
        fwd = -k[:, None] * np.ones((1, CROP))
        left = -k[None, :] * np.ones((CROP, 1))
        eff = self.effective_hit()
        c, s = math.cos(self.heading), math.sin(self.heading)
        cx, cy = self._cell(self.x + fwd * c - left * s, self.y + fwd * s + left * c)
        ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
        cx, cy = np.clip(cx, 0, SIZE - 1), np.clip(cy, 0, SIZE - 1)
        hit = eff[cy, cx]
        free = self.free[cy, cx]
        # frontier: seen floor that touches a never-seen cell = where unexplored space begins
        known = (self.free > 0) | (eff > 0)
        unknown_nb = np.zeros_like(free, dtype=bool)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            unknown_nb |= ~known[np.clip(cy + dy, 0, SIZE - 1), np.clip(cx + dx, 0, SIZE - 1)]
        frontier = ((free > 0) & (hit == 0) & unknown_nb) | ((self.exit[cy, cx] > 0) & (hit == 0))
        out = free | ((hit == 1) << 1) | ((hit == 2) << 2) | ((hit == 3) << 3) | (self.visit[cy, cx] << 4) | (frontier.astype(np.uint8) << 5)
        return (out * ok).astype(np.uint8)

    def wide_packed(self) -> np.ndarray:
        """(WIDE, WIDE) uint8 zoomed-out ego-centric view, one bit per layer, each true if ANY of the
        4x4 fine cells under the coarse cell has it: 1 seen floor, 2 visited, 4 frontier, 8 door,
        16 locked door, 32 wall. Lets the network see unexplored space well beyond the fine crop."""
        sub = (np.arange(4) - 1.5) * CELL
        k = (np.arange(WIDE) - WIDE / 2 + 0.5) * WIDE_CELL
        fwd = (-k[:, None, None, None] + sub[None, None, :, None]) * np.ones((WIDE, WIDE, 4, 4))
        left = (-k[None, :, None, None] + sub[None, None, None, :]) * np.ones((WIDE, WIDE, 4, 4))
        eff = self.effective_hit()
        c, s = math.cos(self.heading), math.sin(self.heading)
        cx, cy = self._cell(self.x + fwd * c - left * s, self.y + fwd * s + left * c)
        ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
        cx, cy = np.clip(cx, 0, SIZE - 1), np.clip(cy, 0, SIZE - 1)
        free = (self.free[cy, cx] > 0) & ok
        hit = eff[cy, cx] * ok
        known = (self.free > 0) | (eff > 0)
        unknown_nb = np.zeros_like(free)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            unknown_nb |= ~known[np.clip(cy + dy, 0, SIZE - 1), np.clip(cx + dx, 0, SIZE - 1)]
        frontier = (free & (hit == 0) & unknown_nb) | ((self.exit[cy, cx] > 0) & ok & (hit == 0))
        anyf = lambda a: a.any(axis=(2, 3)).astype(np.uint8)  # noqa: E731
        return (anyf(free) | (anyf((self.visit[cy, cx] > 0) & ok) << 1) | (anyf(frontier) << 2)
                | (anyf(hit == 2) << 3) | (anyf(hit == 3) << 4) | (anyf(hit == 1) << 5)).astype(np.uint8)

    def crop(self) -> np.ndarray:
        """(CHANNELS, CROP, CROP) float32 ego-centric map: row 0 = ahead, column 0 = left."""
        k = (np.arange(CROP) - CROP / 2 + 0.5) * CROP_CELL
        fwd = -k[:, None] * np.ones((1, CROP))          # rows: ahead (top) to behind (bottom)
        left = -k[None, :] * np.ones((CROP, 1))         # cols: left to right
        eff = self.effective_hit()
        c, s = math.cos(self.heading), math.sin(self.heading)
        wx = self.x + fwd * c - left * s
        wy = self.y + fwd * s + left * c
        cx, cy = self._cell(wx, wy)
        ok = (cx >= 0) & (cx < SIZE) & (cy >= 0) & (cy < SIZE)
        cx, cy = np.clip(cx, 0, SIZE - 1), np.clip(cy, 0, SIZE - 1)
        out = np.zeros((CHANNELS, CROP, CROP), np.float32)
        free = self.free[cy, cx] * ok
        hit = eff[cy, cx] * ok
        out[0] = free
        out[1] = hit == 1
        out[2] = hit == 2
        out[3] = hit == 3
        out[4] = self.visit[cy, cx] * ok
        out[5] = (free == 0) & (hit == 0)
        return out
