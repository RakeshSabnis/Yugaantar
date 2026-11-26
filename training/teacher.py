"""Scripted teacher (TRAINING ONLY; never imported by submission/).

It supplies the demonstrations the network imitates before PPO fine-tuning. To keep those
demonstrations learnable it deliberately plays under the same partial observability as the agent:
it plans on the occupancy grid that `submission.mapper.Mapper` builds from the rays, and it reacts
only to entities in the observation. Its one privilege is the true pose, so its map never drifts.

Priorities follow the scoring order: survive -> seen key -> locked door (with a key) -> nearest
unexplored space / unopened door. Walking within one cell of every free cell also finds the exit,
which no ray or entity reveals.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from submission.mapper import DOOR, EXIT, FREE, G, KEY, LOCKED, RES, VISITED, WALL, Mapper

FWD, BACK, LEFT, RIGHT, TL, TR, ATK, USE, NEXTW = range(9)
COVER = 3  # walking within this many cells of a free cell counts as having covered it
N8 = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))


def _dilate(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for ax in (0, 1):
        out |= np.roll(a, 1, ax) | np.roll(a, -1, ax)
    return out


def _dilate8(a: np.ndarray) -> np.ndarray:
    b = a | np.roll(a, 1, 0) | np.roll(a, -1, 0)
    return b | np.roll(b, 1, 1) | np.roll(b, -1, 1)


class Teacher:
    def __init__(self, rng: np.random.Generator | None = None):
        self.map = Mapper()
        self.rng = rng or np.random.default_rng(0)

    def reset(self, env) -> None:
        st = env.privileged_state()
        self.x0, self.y0, self.a0 = st["x"], st["y"], math.radians(st["angle"])
        self.map.reset()
        self.covered = np.zeros((G, G), bool)
        self.used = np.zeros((G, G), bool)  # doors we have already opened once (not goals any more)
        self.blocked = np.zeros((G, G), bool)  # locked doors that refused to open, cells we got stuck on
        self.path: list[tuple[int, int]] = []
        self.goal_kind = ""
        self.replan_in = 0
        self.use_cd = 0
        self.use_wait = 0
        self.last_pos = (0.0, 0.0)
        self.still = 0
        self.unstick = 0
        self.unstick_act = None
        self.keys_held = 0
        self.strafe_dir = 1
        self.t = 0

    # ------------------------------------------------------------------ planning
    def _plan(self, start, has_key: bool, keep=None):
        L = self.map.layers
        free = L[FREE].astype(bool)
        wall = L[WALL].astype(bool)
        door = L[DOOR].astype(bool)
        locked = L[LOCKED].astype(bool)
        known = free | wall | door | locked
        passable = (free | door | (locked & has_key)) & ~wall & ~(self.blocked & ~free)
        near_wall = _dilate8(wall)
        # unknown cells right next to a known wall are just gaps between ray hits, not openings
        frontier = free & _dilate(~known & ~near_wall)
        uncovered = free & ~self.covered
        key_goal = L[KEY].astype(bool)
        exit_goal = L[EXIT].astype(bool) & free
        lock_goal = locked & has_key & ~self.blocked & ~self.used
        door_goal = door & ~self.blocked & ~self.used
        kind_of = lambda x, y: ("exit" if exit_goal[x, y] else "key" if key_goal[x, y] else "locked" if lock_goal[x, y] else "door" if door_goal[x, y]
                                else "frontier" if frontier[x, y] else "cover" if uncovered[x, y] else None)
        # Commit to the previous target while it is still worth going to (avoids dithering between
        # two equally near targets); keys and usable locked doors always pre-empt.
        if keep is not None and (exit_goal.any() or key_goal.any() or lock_goal.any() or kind_of(*keep) is None):
            keep = None
        dist = {start: 0.0}
        prev = {}
        heap = [(0.0, start)]
        best = {}  # goal kind -> (cost, cell)
        bias = {"exit": 0.0, "key": 0.0, "locked": 0.0, "door": 12.0, "frontier": 0.0, "cover": 400.0}  # sweep rooms only when nothing new is left to see
        while heap:
            d, (x, y) = heapq.heappop(heap)
            if d > dist.get((x, y), 1e18):
                continue
            if d > 0:
                kind = kind_of(x, y)
                if keep is not None:
                    if (x, y) == keep:
                        best = {kind: (d, (x, y))}
                        break
                    kind = None
                if kind and kind not in best:
                    best[kind] = (d, (x, y))
                    if kind in ("exit", "key", "locked"):
                        break
                if keep is None and (d > 1500 or best and d > min(c + bias[k] for k, (c, _) in best.items()) + 24):
                    break
                if (door[x, y] or locked[x, y]) and not self.used[x, y]:
                    continue  # plan up to an unopened door, not through it
            for dx, dy, c in N8:
                nx, ny = (x + dx) % G, (y + dy) % G
                if not passable[nx, ny]:
                    continue
                if dx and dy and not (passable[(x + dx) % G, y] and passable[x, (y + dy) % G]):
                    continue  # no corner cutting
                nd = d + c * (1.0 + 2.5 * near_wall[nx, ny])
                if nd < dist.get((nx, ny), 1e18):
                    dist[(nx, ny)] = nd
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(heap, (nd, (nx, ny)))
        if not best:
            return (self._plan(start, has_key) if keep is not None else ([], ""))
        if "exit" in best:
            kind = "exit"
        elif "key" in best:
            kind = "key"
        elif "locked" in best:
            kind = "locked"
        else:
            kind = min(best, key=lambda k: best[k][0] + bias[k])
        cell = best[kind][1]
        path = [cell]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        return path, kind

    # ------------------------------------------------------------------ acting
    def act(self, obs: dict, env) -> np.ndarray:
        st = env.privileged_state()
        dx, dy = st["x"] - self.x0, st["y"] - self.y0
        c0, s0 = math.cos(self.a0), math.sin(self.a0)
        m = self.map
        m.set_pose(dx * c0 + dy * s0, -dx * s0 + dy * c0, math.radians(st["angle"]) - self.a0)
        m.update(obs, dead_reckon=False)
        px, py = (int(v) for v in m.cell(m.x, m.y))
        r = np.arange(-COVER, COVER + 1)
        self.covered[((px + r) % G)[:, None], ((py + r) % G)[None, :]] = True
        a = np.zeros(9, np.int8)
        self.t += 1
        self.use_cd = max(0, self.use_cd - 1)
        vit = obs["vitals"]
        keys = int(np.sum(vit[8:14]))
        if keys != self.keys_held:
            self.keys_held = keys
            self.blocked[:] = False
            self.replan_in = 0
        rays_d = np.asarray(obs["rays"]["distance"]) * 1024.0
        rays_t = np.asarray(obs["rays"]["hit_type"])

        # ---- stuck handling
        moved = math.hypot(m.x - self.last_pos[0], m.y - self.last_pos[1])
        self.last_pos = (m.x, m.y)
        if self.unstick > 0:
            self.unstick -= 1
            return self.unstick_act.copy()

        # ---- combat
        ents = obs["visible_entities"]
        et = np.asarray(ents["entity_type"])
        feat = np.asarray(ents["features"])
        enemy = (np.asarray(ents["mask"]) > 0) & (et >= 1) & (et <= 3)
        has_ammo = vit[3] > 0
        if enemy.any() and has_ammo:
            idx = np.flatnonzero(enemy)
            near = [j for j in idx if feat[j, 0] * 1024.0 < 160.0]
            hits = [j for j in idx if et[j] == 2]
            i = int((near or hits or list(idx))[0])  # point-blank threat, else hitscanners, else nearest
            d = float(feat[i, 0]) * 1024.0
            ang = math.atan2(float(feat[i, 1]), float(feat[i, 2]))
            if d < 900.0 or d < 1400 and et[i] != 1:
                tol = math.atan2(18.0, max(d, 1.0)) + math.radians(1.5)
                if abs(ang) > math.radians(7.5):
                    a[TL if ang > 0 else TR] = 1
                elif abs(ang) > tol:
                    a[LEFT if ang > 0 else RIGHT] = 1  # fine aim by side-stepping
                if abs(ang) <= max(tol, math.radians(3.0)):
                    a[ATK] = 1
                    if self.t % 12 == 0:
                        self.strafe_dir = -self.strafe_dir
                    if not a[LEFT] and not a[RIGHT] and d > 150 and et[i] != 1:
                        a[LEFT if self.strafe_dir > 0 else RIGHT] = 1  # dodge while firing
                if et[i] == 1 and d < 130:
                    a[BACK] = 1  # keep melee monsters at arm's length
                self.still = 0
                self.replan_in = 0
                return a

        # ---- something is attacking from outside the field of view: turn to face it
        if has_ammo and not enemy.any():
            er = np.flatnonzero((rays_t == 4) & (rays_d < 500.0))
            if len(er):
                k = int(er[np.argmin(rays_d[er])])
                a[TL if k < 32 else TR] = 1
                return a
        # ---- pickups in view: health when hurt, ammo when low, weapons, armour, keys
        hp, ammo = float(vit[0]), float(vit[3])
        want = (np.asarray(ents["mask"]) > 0) & (feat[:, 0] * 1024.0 < 420.0) & (
            ((et == 5) & (hp < 80)) | ((et == 7) & (ammo < 60)) | (et == 8) | ((et == 6) & (vit[1] < 50)) | (et == 9))
        if want.any():
            i = int(np.flatnonzero(want)[0])
            ang = math.atan2(float(feat[i, 1]), float(feat[i, 2]))
            if abs(ang) > math.radians(9):
                a[TL if ang > 0 else TR] = 1
            if abs(ang) < math.radians(30):
                a[FWD] = 1
            self.still = self.still + 1 if moved < 1.0 else 0
            if self.still < 12:
                return a
            a[:] = 0  # can't reach it (ledge, monster in the way): fall through to navigation

        # ---- door directly ahead: press USE once, then wait for it
        ahead = [0, 1, 63]
        door_ahead = [k for k in ahead if rays_t[k] in (2, 3) and rays_d[k] < 62.0]
        if door_ahead:
            locked = any(rays_t[k] == 3 for k in door_ahead)
            if not locked or keys:
                if self.use_cd == 0:
                    if self.use_wait >= 2:  # tried and it stayed shut (wrong key colour): give up on it
                        hx, hy = (int(v) for v in m.cell(m.x + math.cos(m.h) * (rays_d[0] + 10), m.y + math.sin(m.h) * (rays_d[0] + 10)))
                        for ox in range(-6, 7):
                            for oy in range(-6, 7):
                                if m.layers[LOCKED, (hx + ox) % G, (hy + oy) % G]:
                                    self.blocked[(hx + ox) % G, (hy + oy) % G] = True
                        self.use_wait = 0
                        self.replan_in = 0
                    else:
                        a[USE] = 1
                        hx, hy = (int(v) for v in m.cell(m.x + math.cos(m.h) * (rays_d[0] + 10), m.y + math.sin(m.h) * (rays_d[0] + 10)))
                        for ox in range(-5, 6):
                            for oy in range(-5, 6):
                                self.used[(hx + ox) % G, (hy + oy) % G] = True
                        self.use_cd = 14
                        self.use_wait += 1
                        return a
                else:
                    return a  # wait for the door
        else:
            self.use_wait = 0

        # ---- navigation
        if self.replan_in <= 0 or not self.path:
            keep = self.path[-1] if self.path and self.goal_kind in ("frontier", "cover", "door") else None
            self.path, self.goal_kind = self._plan((px, py), keys > 0, keep)
            self.replan_in = 4
        self.replan_in -= 1
        if not self.path:
            a[TL] = 1  # nothing left to do: look around
            return a
        # drop waypoints already reached, then aim a few cells ahead
        k = min(range(len(self.path)), key=lambda j: (self.path[j][0] - px) ** 2 + (self.path[j][1] - py) ** 2)
        self.path = self.path[k:]
        tgt = self.path[min(5, len(self.path) - 1)]
        if len(self.path) <= 1:
            self.replan_in = 0
        tx = (((tgt[0] - px + G // 2) % G) - G // 2 + 0.5) * RES + (math.floor(m.x / RES) * RES) - m.x
        ty = (((tgt[1] - py + G // 2) % G) - G // 2 + 0.5) * RES + (math.floor(m.y / RES) * RES) - m.y
        err = (math.atan2(ty, tx) - m.h + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > math.radians(9):
            a[TL if err > 0 else TR] = 1
        elif abs(err) > math.radians(3):
            a[LEFT if err > 0 else RIGHT] = 1
        if abs(err) < math.radians(35):
            a[FWD] = 1
        # keep off the walls
        if rays_d[16] < 22 and rays_d[48] > 30:
            a[RIGHT], a[LEFT] = 1, 0
        elif rays_d[48] < 22 and rays_d[16] > 30:
            a[LEFT], a[RIGHT] = 1, 0

        self.still = self.still + 1 if (moved < 1.0 and a[FWD]) else 0
        if self.still > 10:
            self.still = 0
            if len(self.path) > 1:
                bx, by = self.path[min(3, len(self.path) - 1)]
                self.blocked[bx, by] = True
            self.unstick = int(self.rng.integers(3, 7))
            u = np.zeros(9, np.int8)
            u[BACK] = 1
            u[LEFT if self.rng.random() < 0.5 else RIGHT] = 1
            self.unstick_act = u
            self.replan_in = 0
        return a
