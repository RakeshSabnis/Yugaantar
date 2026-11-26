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


MAP_VIEW = 72      # cells shown around the agent (72 x 64 = 4608 map units)
MAP_PX = 6         # pixels per cell
COLORS = {"unknown": (40, 40, 46), "free": (150, 150, 140), "visited": (90, 140, 210), "wall": (240, 240, 240),
          "door": (200, 140, 60), "locked": (230, 50, 50), "agent": (60, 230, 90)}


class MapView:
    """One window: the game view on the left, the agent's own SLAM map on the right."""

    def __init__(self):
        import os

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pg = pygame
        pygame.init()
        size = MAP_VIEW * MAP_PX
        self.mx = GAME_W + 8  # map panel x offset
        self.screen = pygame.display.set_mode((GAME_W + 8 + size, max(GAME_H, size) + 28))
        pygame.display.set_caption("DOOM agent: game view + SLAM map")
        self.font = pygame.font.SysFont(None, 22)

    def draw(self, slam, info: str, frame=None) -> bool:
        import math

        import numpy as np
        from slam import CELL, SIZE

        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT:
                return False
        cx = int(math.floor(slam.x / CELL)) + SIZE // 2
        cy = int(math.floor(slam.y / CELL)) + SIZE // 2
        half = MAP_VIEW // 2
        ys, xs = np.mgrid[cy + half - 1:cy - half - 1:-1, cx - half:cx + half]  # north up
        ok = (xs >= 0) & (xs < SIZE) & (ys >= 0) & (ys < SIZE)
        xs, ys = np.clip(xs, 0, SIZE - 1), np.clip(ys, 0, SIZE - 1)
        free, hit, vis = slam.free[ys, xs] * ok, slam.hit[ys, xs] * ok, slam.visit[ys, xs] * ok
        img = np.empty((MAP_VIEW, MAP_VIEW, 3), np.uint8)
        img[:] = COLORS["unknown"]
        img[free > 0] = COLORS["free"]
        img[vis > 0] = COLORS["visited"]
        img[hit == 1] = COLORS["wall"]
        img[hit == 2] = COLORS["door"]
        img[hit == 3] = COLORS["locked"]
        surf = pg.surfarray.make_surface(np.repeat(np.repeat(img, MAP_PX, 0), MAP_PX, 1).transpose(1, 0, 2))
        self.screen.fill((0, 0, 0))
        if frame is not None and frame.ndim == 3:
            self.screen.blit(pg.surfarray.make_surface(np.ascontiguousarray(frame.transpose(1, 0, 2))), (0, 0))
        self.screen.blit(surf, (self.mx, 0))
        # agent: dot + heading line at the centre cell
        px = self.mx + half * MAP_PX + MAP_PX // 2
        py = (half - 1) * MAP_PX + MAP_PX // 2
        hx, hy = px + 18 * math.cos(slam.heading), py - 18 * math.sin(slam.heading)
        pg.draw.line(self.screen, COLORS["agent"], (px, py), (hx, hy), 3)
        pg.draw.circle(self.screen, COLORS["agent"], (px, py), 5)
        self.screen.blit(self.font.render(info, True, (230, 230, 230)), (6, max(GAME_H, MAP_VIEW * MAP_PX) + 5))
        pg.display.flip()
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="agent", choices=["agent", "teacher"])
    ap.add_argument("--weights", default=str(ROOT / "runs" / "dagger_slam" / "best.pt"),
                    help="checkpoint to play (default: best map-equipped agent so far)")
    ap.add_argument("--map", default=None, help="file.wad or file.wad:MAP01 (default: random held-out map)")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real game speed")
    ap.add_argument("--no-map", action="store_true", help="don't open the SLAM map window")
    ap.add_argument("--save-map", default=None, help="save the final map window as a PNG")
    a = ap.parse_args()
    if a.policy == "agent" and not Path(a.weights).exists():
        print(f"no checkpoint at {a.weights} yet: showing the teacher instead")
        a.policy = "teacher"

    from doom_env import make_env

    from actions import to_buttons
    from scorer import Tracker

    if a.map is None:
        items = json.loads((ROOT / "maps_gen" / "heldout" / "index.json").read_text())
        item = random.choice(items)
        wad, mapname, ticks = item["path"], "MAP01", item["ticks"]
    else:
        wad, _, mapname = a.map.partition(":")
        mapname = mapname or "MAP01"
        ticks = 4200
    print(f"map {wad}:{mapname}")
    grid_path = Path(wad).with_suffix(".txt")
    if grid_path.exists():
        print(grid_path.read_text())
    env = make_env(wad, mapname, episode_ticks=ticks, render_buffers=not a.no_map)
    obs, _ = env.reset(seed=0)
    tracker = Tracker(env.privileged_state())
    if a.policy == "teacher":
        from teacher import Teacher, door_open_fn

        grid = grid_path.read_text()
        t = Teacher(grid)
        door_open = door_open_fn(env, grid)
        step = lambda o: to_buttons(t.act(o, env.privileged_state(), door_open))  # noqa: E731
    else:
        from agent import DoomAgent

        agent = DoomAgent(env.observation_space, env.action_space, "cpu", weights_path=a.weights)
        agent.reset()
        step = agent.act
    from slam import Slam

    view = None if a.no_map else MapView()
    slam = Slam()  # display copy, built exactly like the agent's: from observations + its own actions
    slam.observe(obs)
    dt = 4 / 35 / a.speed
    k = 0
    try:
        while True:
            t0 = time.time()
            buttons = step(obs)
            slam.apply_action(buttons)
            obs, r, term, trunc, info = env.step(buttons)
            slam.observe(obs)
            tracker.update(env.privileged_state())
            k += 1
            if view is not None:
                v = obs["vitals"]
                keys = "".join(c for c, i in (("B", 8), ("Y", 9), ("R", 10)) if v[i] > 0 or v[i + 3] > 0)
                if not view.draw(slam, f"step {k}  health {int(v[0])}  ammo {int(v[3])}  keys {keys or '-'}", obs.get("rgb")):
                    view = None
            if term or trunc:
                break
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    if a.save_map and view is not None:
        view.pg.image.save(view.screen, a.save_map)
    res = tracker.result(Path(wad).name, 0).as_dict()
    print({k: res[k] for k in ("exit", "dead", "keys", "doors_opened", "locks_opened", "areas", "kills", "health", "score")})
    env.close()


if __name__ == "__main__":
    main()
