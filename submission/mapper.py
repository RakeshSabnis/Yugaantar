"""Egocentric spatial memory built only from the official observation dict.

The observation has no position or heading, so the agent keeps its own pose estimate by dead
reckoning (heading from its own turn buttons, position from the velocity in `vitals`) and paints
what the 64 rays reveal into an occupancy grid anchored at the spawn point. The network never sees
that grid directly: it gets two egocentric crops (forward = up), a fine one covering ray range and a
coarse one covering +-2048 units, so the memory is translation- and rotation-invariant across maps.

This module is preprocessing only: it never chooses an action. During training the teacher uses the
same class with the true pose (`set_pose`), which is what makes its behaviour imitable.
"""

from __future__ import annotations

import math

import numpy as np

RES = 16.0  # map units per cell: fine enough that a 64-unit corridor stays open at any angle
G = 512  # grid cells per side (8192 units; the spawn is at the centre, indices wrap around)
RAY_RANGE = 1024.0
NUM_RAYS = 64
# layers
FREE, WALL, DOOR, LOCKED, VISITED, KEY, EXIT = range(7)
NUM_LAYERS = 7
CROP = 32  # crop side in pixels
FINE = 2  # fine crop: 2x2 cells (32 units) per pixel, +-512 units
COARSE = 8  # coarse crop: 8x8 cells (128 units) per pixel, +-2048 units

# Keyboard turning in the vanilla engine, per game tic: slow for the first SLOW_TICS consecutive tics
# that either turn key is held, fast afterwards. Switching direction does not reset the count, both
# keys together cancel, and the first DEAD_TICS tics of an episode ignore turning. Measured on
# ViZDoom 1.3.3; the dead-reckoned heading then matches the engine exactly.
TURN_SLOW, TURN_FAST = math.radians(1.7578125), math.radians(3.515625)
SLOW_TICS, DEAD_TICS, FRAME_SKIP = 5, 7, 4
# Displacement over one step from the velocity (units/tic) observed before and after it.
DISP_PREV, DISP_END = 1.53, 2.89

# Exit pads are not in the rays or the entity list; the only cue is the frame. The exit floor is
# saturated red, and the depth buffer places each such pixel: forward = depth * DEPTH_UNITS, and a
# floor pixel at image row y satisfies forward * (y - HORIZON) ~= FLOOR_K (eye height x focal length).
DEPTH_UNITS, HORIZON, FLOOR_K, FLOOR_TOL = 7.6, 60.0, 3800.0, 1000.0
_PIX_ROWS = np.arange(62, 120, dtype=np.float64)[:, None]
_PIX_LEFT = ((79.5 - np.arange(160)) / 80.0)[None, :]

_RAY_ANG = np.arange(NUM_RAYS) * (2.0 * np.pi / NUM_RAYS)
_SAMPLES = np.arange(RES / 2, RAY_RANGE, RES / 2)  # along-ray sample distances


def _offsets(n: int, step: float) -> tuple[np.ndarray, np.ndarray]:
    """(forward, left) offsets in map units of an n x n image whose top row is farthest ahead."""
    idx = (np.arange(n) - (n - 1) / 2.0) * step
    fwd = -idx[:, None] + np.zeros((1, n))
    left = -idx[None, :] + np.zeros((n, 1))
    return fwd, left


_FINE = _offsets(CROP * FINE, RES)
_COARSE = _offsets(CROP * COARSE, RES)


class Mapper:
    def __init__(self):
        self.layers = np.zeros((NUM_LAYERS, G, G), np.uint8)
        self.reset()

    def reset(self) -> None:
        self.layers[:] = 0
        self.x = self.y = self.h = 0.0
        self.turn_held = 0
        self.tic = 0
        self.prev_vel = np.zeros(2)  # world frame
        self.steps = 0

    # ------------------------------------------------------------------ pose
    def set_pose(self, x: float, y: float, h: float) -> None:
        """Training only: overwrite the estimate with ground truth (spawn-relative)."""
        self.x, self.y, self.h = x, y, h

    def _dead_reckon(self, vitals: np.ndarray, prev_action) -> None:
        if prev_action is not None:
            left, right = int(prev_action[4]), int(prev_action[5])
            for _ in range(FRAME_SKIP):
                self.tic += 1
                if left or right:
                    self.turn_held += 1
                    if self.tic > DEAD_TICS:
                        self.h += (left - right) * (TURN_SLOW if self.turn_held <= SLOW_TICS else TURN_FAST)
                else:
                    self.turn_held = 0
        c, s = math.cos(self.h), math.sin(self.h)
        f, r = float(vitals[14]), float(vitals[15])
        vel = np.array([f * c + r * s, f * s - r * c])
        if prev_action is not None:
            dx, dy = DISP_PREV * self.prev_vel + DISP_END * vel
            self.x += float(dx)
            self.y += float(dy)
        self.prev_vel = vel

    # ------------------------------------------------------------------ update
    def cell(self, x, y):
        return (np.floor(np.asarray(x) / RES).astype(np.int64) + G // 2) % G, (np.floor(np.asarray(y) / RES).astype(np.int64) + G // 2) % G

    def update(self, obs: dict, prev_action=None, dead_reckon: bool = True) -> None:
        if dead_reckon:
            self._dead_reckon(obs["vitals"], prev_action)
        L = self.layers
        dist = np.asarray(obs["rays"]["distance"], np.float64) * RAY_RANGE
        hit = np.asarray(obs["rays"]["hit_type"])
        ang = self.h + _RAY_ANG
        ca, sa = np.cos(ang), np.sin(ang)
        # free space along every ray, up to just before the hit
        t = _SAMPLES[None, :]
        ok = t < (dist[:, None] - RES * 0.35)
        ix, iy = self.cell(self.x + ca[:, None] * t, self.y + sa[:, None] * t)
        ix, iy = ix[ok], iy[ok]
        L[FREE, ix, iy] = 1
        L[DOOR, ix, iy] = 0  # an opened door reads as free until it is seen closed again
        L[LOCKED, ix, iy] = 0
        L[WALL, ix, iy] = 0
        # what stopped the ray
        solid = (hit >= 1) & (hit <= 3) & (dist < RAY_RANGE - 1.0)
        if solid.any():
            hx, hy = self.cell(self.x + ca[solid] * (dist[solid] + RES * 0.3), self.y + sa[solid] * (dist[solid] + RES * 0.3))
            kind = hit[solid]
            for code, layer in ((1, WALL), (2, DOOR), (3, LOCKED)):
                m = kind == code
                if m.any():
                    L[:4, hx[m], hy[m]] = 0
                    L[layer, hx[m], hy[m]] = 1
        px, py = self.cell(self.x, self.y)
        L[VISITED, px, py] = 1
        L[FREE, px, py] = 1
        L[KEY, ((px + np.arange(-2, 3)) % G)[:, None], ((py + np.arange(-2, 3)) % G)[None, :]] = 0  # picked up / walked over
        ents = obs["visible_entities"]
        keys = (np.asarray(ents["mask"]) > 0) & (np.asarray(ents["entity_type"]) == 9)
        if keys.any():
            f = np.asarray(ents["features"], np.float64)[keys]
            d = f[:, 0] * RAY_RANGE
            c, s = math.cos(self.h), math.sin(self.h)
            ex, ey = d * f[:, 2], d * f[:, 1]  # forward, left
            kx, ky = self.cell(self.x + ex * c - ey * s, self.y + ex * s + ey * c)
            L[KEY, kx, ky] = 1
        L[EXIT, ((px + np.arange(-2, 3)) % G)[:, None], ((py + np.arange(-2, 3)) % G)[None, :]] = 0  # stood there: not an exit
        rgb = obs.get("rgb")
        if rgb is not None:
            lo = np.asarray(rgb)[62:]
            r = lo[..., 0].astype(np.int16)
            red = (r > 80) & (lo[..., 1] < r * 0.35) & (lo[..., 2] < r * 0.35)
            if red.any():
                fwd = np.asarray(obs["depth"])[62:].astype(np.float64) * DEPTH_UNITS
                red &= np.abs(fwd * (_PIX_ROWS - HORIZON) - FLOOR_K) < FLOOR_TOL
                if red.sum() >= 3:
                    f = fwd[red]
                    lf = (fwd * _PIX_LEFT)[red]
                    c, s = math.cos(self.h), math.sin(self.h)
                    ex, ey = self.cell(self.x + f * c - lf * s, self.y + f * s + lf * c)
                    L[EXIT, ex, ey] = 1
        self.steps += 1

    # ------------------------------------------------------------------ network input
    def _sample(self, offs) -> np.ndarray:
        fwd, left = offs
        c, s = math.cos(self.h), math.sin(self.h)
        ix, iy = self.cell(self.x + fwd * c - left * s, self.y + fwd * s + left * c)
        return self.layers[:, ix, iy]

    def crops(self) -> np.ndarray:
        """(2 * NUM_LAYERS, CROP, CROP) float32: fine crop then coarse crop, forward = up."""
        fine = self._sample(_FINE).reshape(NUM_LAYERS, CROP, FINE, CROP, FINE).max(axis=(2, 4))
        coarse = self._sample(_COARSE).reshape(NUM_LAYERS, CROP, COARSE, CROP, COARSE).max(axis=(2, 4))
        return np.concatenate([fine, coarse]).astype(np.float32)
