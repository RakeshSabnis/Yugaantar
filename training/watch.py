"""Watch a policy play in a visible game window at real game speed.

    uv run python training/watch.py                                   # best imitation checkpoint, random held-out map
    uv run python training/watch.py --weights runs/dagger1/best.pt --map maps_gen/heldout/t3_100004.wad
    uv run python training/watch.py --policy teacher --map maps_gen/heldout/t3_100004.wad
    uv run python training/watch.py --map /path/to/freedoom2.wad:MAP01

Close the window or press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import vizdoom as vd  # noqa: E402

# Render the game at a watchable size into our own window (the engine's window stays hidden). The agent
# never reads the rgb/depth buffers, so this changes nothing about what it sees.
_set_res = vd.DoomGame.set_screen_resolution
vd.DoomGame.set_screen_resolution = lambda self, r: _set_res(self, vd.ScreenResolution.RES_640X480)
GAME_W, GAME_H = 640, 480
PANEL_H = 4 * 24 + 10  # status panel under the game view


MAP_SIZE = 680     # map panel size in pixels
MAP_SPAN = 3400.0  # map units shown across the panel
BG = (11, 15, 20)
GRID = (24, 32, 42)
FREE = (38, 84, 96)
WALL_PT = (214, 226, 235)
DOOR_PT = (255, 176, 64)
LOCK_PT = (255, 72, 88)
TRAJ = (64, 255, 160)
KEYFRAME = (80, 150, 255)
SCAN = (0, 210, 255)
AGENT = (255, 255, 255)


class MapView:
    """One window: the game view on the left, the agent's own SLAM map on the right.

    Drawn like a modern SLAM viewer: a point cloud of everything the rays have hit (walls, doors,
    locked doors), the explored free space, the trajectory with keyframes, the live scan and the
    agent's pose with its field of view. All of it comes from the agent's own observations and actions.
    """

    def __init__(self):
        import os

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pg = pygame
        pygame.init()
        self.mx = GAME_W + 8  # map panel x offset
        self.screen = pygame.display.set_mode((GAME_W + 8 + MAP_SIZE, max(GAME_H, MAP_SIZE) + PANEL_H))
        pygame.display.set_caption("DOOM agent: game view + SLAM map")
        self.font = pygame.font.SysFont("menlo,monaco,couriernew", 15)
        self.small = pygame.font.SysFont("menlo,monaco,couriernew", 14)
        self.traj: list[tuple[float, float]] = []
        self.pts = {1: set(), 2: set(), 3: set()}  # hit points snapped to 8 units, by type

    def reset(self) -> None:
        self.traj = []
        self.pts = {1: set(), 2: set(), 3: set()}

    def _accumulate(self, slam, obs) -> list[tuple[float, float]]:
        import math

        import numpy as np

        self.traj.append((slam.x, slam.y))
        d = np.asarray(obs["rays"]["distance"], np.float64) * 1024.0
        ht = np.asarray(obs["rays"]["hit_type"])
        ang = slam.heading + np.arange(64) * (2 * math.pi / 64)
        hx, hy = slam.x + np.cos(ang) * d, slam.y + np.sin(ang) * d
        for k in range(64):
            t = int(ht[k])
            if t in self.pts and d[k] < 1023:
                self.pts[t].add((round(hx[k] / 8) * 8, round(hy[k] / 8) * 8))
        return [(hx[k], hy[k]) for k in range(64) if d[k] < 1023]

    def draw(self, slam, info, frame=None, obs=None) -> bool:
        import math

        import numpy as np
        from slam import CELL, SIZE

        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT:
                return False
        scan = self._accumulate(slam, obs) if obs is not None else []
        S, sc = MAP_SIZE, MAP_SIZE / MAP_SPAN
        cxw, cyw = slam.x, slam.y

        def to_px(x, y):
            return (S / 2 + (x - cxw) * sc, S / 2 - (y - cyw) * sc)

        panel = pg.Surface((S, S), 0, 24)  # opaque: no alpha channel
        panel.fill(BG)
        # explored free space: occupancy grid, smoothly scaled, drawn softly under everything else
        half = int(MAP_SPAN / CELL / 2) + 1
        ccx, ccy = int(math.floor(cxw / CELL)) + SIZE // 2, int(math.floor(cyw / CELL)) + SIZE // 2
        ys, xs = np.mgrid[ccy + half:ccy - half - 1:-1, ccx - half:ccx + half + 1]
        ok = (xs >= 0) & (xs < SIZE) & (ys >= 0) & (ys < SIZE)
        free = slam.free[np.clip(ys, 0, SIZE - 1), np.clip(xs, 0, SIZE - 1)] * ok
        img = np.zeros(free.shape + (3,), np.uint8)
        img[free > 0] = FREE
        img[free == 0] = BG
        surf = pg.surfarray.make_surface(img.transpose(1, 0, 2))
        wpx = int(round(free.shape[1] * CELL * sc))
        surf = pg.transform.smoothscale(surf, (wpx, wpx))
        ox, oy = to_px((ccx - half - SIZE // 2) * CELL, (ccy + half + 1 - SIZE // 2) * CELL)
        panel.blit(surf, (ox, oy))
        # grid every 256 map units
        g0x, g0y = math.floor((cxw - MAP_SPAN / 2) / 256) * 256, math.floor((cyw - MAP_SPAN / 2) / 256) * 256
        for n in range(int(MAP_SPAN // 256) + 2):
            x = to_px(g0x + n * 256, 0)[0]
            y = to_px(0, g0y + n * 256)[1]
            pg.draw.line(panel, GRID, (x, 0), (x, S))
            pg.draw.line(panel, GRID, (0, y), (S, y))
        # live scan: faint rays from the agent to this step's hits
        overlay = pg.Surface((S, S), pg.SRCALPHA)
        a0 = to_px(slam.x, slam.y)
        for hx, hy in scan:
            pg.draw.aaline(overlay, SCAN + (38,), a0, to_px(hx, hy))
        # field of view wedge (90 degrees)
        fov = [a0] + [to_px(slam.x + 520 * math.cos(slam.heading + t), slam.y + 520 * math.sin(slam.heading + t))
                      for t in np.linspace(-math.pi / 4, math.pi / 4, 12)]
        pg.draw.polygon(overlay, (255, 255, 255, 22), fov)
        panel.blit(overlay, (0, 0))
        # point cloud of everything the rays have hit
        for t, col, r in ((1, WALL_PT, 1), (2, DOOR_PT, 3), (3, LOCK_PT, 3)):
            for x, y in self.pts[t]:
                px, py = to_px(x, y)
                if 0 <= px < S and 0 <= py < S:
                    if r == 1:
                        panel.fill(col, (int(px), int(py), 2, 2))
                    else:
                        pg.draw.circle(panel, col, (int(px), int(py)), r)
        # trajectory with keyframes
        if len(self.traj) > 1:
            pg.draw.lines(panel, TRAJ, False, [to_px(x, y) for x, y in self.traj], 2)
            for x, y in self.traj[::20]:
                px, py = to_px(x, y)
                pg.draw.rect(panel, KEYFRAME, (px - 3, py - 3, 7, 7), 1)
        # agent pose: arrowhead
        h = slam.heading
        tip = to_px(slam.x + 110 * math.cos(h), slam.y + 110 * math.sin(h))
        l = to_px(slam.x + 60 * math.cos(h + 2.5), slam.y + 60 * math.sin(h + 2.5))
        r_ = to_px(slam.x + 60 * math.cos(h - 2.5), slam.y + 60 * math.sin(h - 2.5))
        pg.draw.polygon(panel, AGENT, [tip, l, r_])
        pg.draw.aalines(panel, TRAJ, True, [tip, l, r_])
        # legend + stats
        n_pts = sum(len(v) for v in self.pts.values())
        panel.blit(self.small.render(f"SLAM  |  map points {n_pts}  keyframes {len(self.traj) // 20 + 1}  pose ({slam.x:.0f}, {slam.y:.0f}, {math.degrees(h) % 360:.0f} deg)", True, (150, 170, 185)), (8, 6))
        lx = 8
        for name, col in (("wall", WALL_PT), ("door", DOOR_PT), ("locked door", LOCK_PT), ("path", TRAJ), ("scan", SCAN), ("free space", (70, 140, 155))):
            pg.draw.circle(panel, col, (lx + 4, S - 12), 4)
            lab = self.small.render(name, True, (150, 170, 185))
            panel.blit(lab, (lx + 12, S - 19))
            lx += 22 + lab.get_width()
        pg.draw.rect(panel, (40, 52, 66), (0, 0, S, S), 1)

        self.screen.fill((6, 8, 11))
        if frame is not None and frame.ndim == 3:
            self.screen.blit(pg.surfarray.make_surface(np.ascontiguousarray(frame.transpose(1, 0, 2))), (0, (max(GAME_H, MAP_SIZE) - GAME_H) // 2))
        self.screen.blit(panel, (self.mx, 0))
        y0 = max(GAME_H, MAP_SIZE) + 8
        for i, (text, color) in enumerate(info):
            self.screen.blit(self.font.render(text, True, color), (8, y0 + 24 * i))
        pg.display.flip()
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="agent", choices=["agent", "teacher"])
    ap.add_argument("--weights", default=str(ROOT / "submission" / "weights" / "policy.pt"),
                    help="checkpoint to play (default: the submitted agent)")
    ap.add_argument("--map", default=None, help="file.wad or file.wad:MAP01 (default: random held-out map)")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real game speed")
    ap.add_argument("--no-map", action="store_true", help="don't open the SLAM map window")
    ap.add_argument("--save-map", default=None, help="save the final map window as a PNG")
    ap.add_argument("--live", action="store_true",
                    help="keep playing: before every game, load the newest checkpoint the training run has written "
                         "(runs/*/last.pt) and play it on a random training map")
    a = ap.parse_args()
    if a.policy == "agent" and not Path(a.weights).exists():
        print(f"no checkpoint at {a.weights} yet: showing the teacher instead")
        a.policy = "teacher"

    view = None if a.no_map else MapView()
    while True:
        weights, label = a.weights, ""
        if a.live:
            cands = sorted((ROOT / "runs").glob("*/last.pt"), key=lambda q: q.stat().st_mtime)
            if cands:
                weights = str(cands[-1])
                label = f"LIVE {cands[-1].parent.name} @ {time.strftime('%H:%M', time.localtime(cands[-1].stat().st_mtime))}  |  "
        try:
            keep_going = play(a, view, weights, label)
        except (RuntimeError, EOFError, OSError) as e:  # e.g. checkpoint being written right now
            print(f"retrying: {e}")
            time.sleep(2)
            continue
        if not a.live or not keep_going:
            break
        time.sleep(1.0)


def play(a, view, weights, label) -> bool:
    """Play one map. Returns False if the user closed the window or pressed Ctrl+C."""
    from doom_env import make_env

    from actions import to_buttons
    from scorer import Tracker

    if a.map is None:
        items = json.loads((ROOT / "maps_gen" / ("train" if a.live else "heldout") / "index.json").read_text())
        item = random.choice(items)
        wad, mapname, ticks = item["path"], "MAP01", item["ticks"]
    else:
        wad, _, mapname = a.map.partition(":")
        mapname = mapname or "MAP01"
        ticks = 4200
    TIER_NAMES = {"t1": "Tier 1: combat", "t2": "Tier 2: exploration", "t3": "Tier 3: keycard", "t4": "Tier 4: gauntlet"}
    stem = Path(wad).stem
    level = f"{label}{TIER_NAMES.get(stem[:2], 'Level')}  |  {stem if mapname == 'MAP01' else stem + ':' + mapname}"
    print(f"map {wad}:{mapname}  ({level})")
    grid_path = Path(wad).with_suffix(".txt")
    if grid_path.exists():
        print(grid_path.read_text())
    totals: dict = {}
    if grid_path.exists():
        g = grid_path.read_text()
        totals = {"keys": sum(g.count(c) for c in "rby"), "locks": sum(g.count(c) for c in "RBY"), "doors": g.count("D"),
                  "monsters": sum(g.count(c) for c in "zscid")}
    env = make_env(wad, mapname, episode_ticks=ticks, render_buffers=not a.no_map)
    obs, _ = env.reset(seed=random.randrange(10**6) if a.live else 0)
    tracker = Tracker(env.privileged_state())
    if a.policy == "teacher":
        from teacher import Teacher, door_open_fn

        grid = grid_path.read_text()
        t = Teacher(grid)
        door_open = door_open_fn(env, grid)
        step = lambda o: to_buttons(t.act(o, env.privileged_state(), door_open))  # noqa: E731
    else:
        from agent import DoomAgent

        agent = DoomAgent(env.observation_space, env.action_space, "cpu", weights_path=weights)
        agent.reset()
        step = agent.act
    from slam import Slam

    if view is not None:
        view.reset()
    slam = Slam()  # display copy, built exactly like the agent's: from observations + its own actions
    slam.observe(obs)
    dt = 4 / 35 / a.speed
    k = 0
    act_ms = 0.0
    stop = False
    try:
        while True:
            t0 = time.time()
            ta = time.perf_counter()
            buttons = step(obs)
            act_ms = 0.9 * act_ms + 0.1 * (time.perf_counter() - ta) * 1000 if k else (time.perf_counter() - ta) * 1000
            slam.apply_action(buttons)
            obs, r, term, trunc, info = env.step(buttons)
            slam.observe(obs)
            tracker.update(env.privileged_state())
            k += 1
            if view is not None:
                v = obs["vitals"]
                pr = env.privileged_state()
                res = tracker.result("", 0)
                keys = "".join(c for c, i in (("B", 8), ("Y", 9), ("R", 10)) if v[i] > 0 or v[i + 3] > 0)
                white, good, bad, dim = (235, 235, 235), (90, 230, 110), (240, 90, 80), (170, 170, 170)
                tot = lambda n: f"/{n}" if n is not None else ""  # noqa: E731
                lines = [
                    (f"{level}   |   step {k}/{ticks // 4}   |   {act_ms:.1f} ms per decision (limit 40 ms)", white if act_ms < 40 else bad),
                    (f"PROGRESS   exit: {'REACHED' if res.exit else 'not yet'}    keycards: {res.keys}{tot(totals.get('keys'))} {('[' + keys + ']') if keys else ''}    "
                     f"locked doors opened: {res.locks_opened}{tot(totals.get('locks'))}    doors opened: {res.doors_opened}{tot(totals.get('doors'))}", good if res.exit else white),
                    (f"EXPLORE / COMBAT   areas explored: {res.areas}    enemies killed: {res.kills}{tot(totals.get('monsters'))}    "
                     f"damage taken: {int(res.damage)}", white),
                    (f"SURVIVAL   health: {int(v[0])}   armor: {int(v[1])}   ammo: {int(v[3])}   {'DEAD' if res.dead else 'alive'}"
                     f"      our score estimate: {res.score():.0f}", bad if res.dead or v[0] < 30 else white),
                ]
                if not view.draw(slam, lines, obs.get("rgb"), obs):
                    view, stop = None, True
                    break
            if term or trunc:
                break
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        stop = True
    if a.save_map and view is not None:
        view.pg.image.save(view.screen, a.save_map)
    res = tracker.result(Path(wad).name, 0).as_dict()
    print({k: res[k] for k in ("exit", "dead", "keys", "doors_opened", "locks_opened", "areas", "kills", "health", "score")})
    env.close()
    return not stop


if __name__ == "__main__":
    main()
