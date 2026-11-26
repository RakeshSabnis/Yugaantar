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

# Show the window (the official wrapper hides it) at a watchable size. Observations are unaffected:
# we never use the rgb/depth buffers.
_set_visible = vd.DoomGame.set_window_visible
_set_res = vd.DoomGame.set_screen_resolution
vd.DoomGame.set_window_visible = lambda self, v: _set_visible(self, True)
vd.DoomGame.set_screen_resolution = lambda self, r: _set_res(self, vd.ScreenResolution.RES_640X480)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="agent", choices=["agent", "teacher"])
    ap.add_argument("--weights", default=str(ROOT / "runs" / "dagger1" / "best.pt"))
    ap.add_argument("--map", default=None, help="file.wad or file.wad:MAP01 (default: random held-out map)")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real game speed")
    a = ap.parse_args()

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
    env = make_env(wad, mapname, episode_ticks=ticks, render_buffers=False)
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
    dt = 4 / 35 / a.speed
    try:
        while True:
            t0 = time.time()
            obs, r, term, trunc, info = env.step(step(obs))
            tracker.update(env.privileged_state())
            if term or trunc:
                break
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    res = tracker.result(Path(wad).name, 0).as_dict()
    print({k: res[k] for k in ("exit", "dead", "keys", "doors_opened", "locks_opened", "areas", "kills", "health", "score")})
    env.close()


if __name__ == "__main__":
    main()
