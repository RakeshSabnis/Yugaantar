"""Run a policy (teacher, random, or a DoomAgent checkpoint) on a map set and score it locally.

    uv run python training/evaluate.py --policy teacher --maps maps_gen/heldout --workers 7
    uv run python training/evaluate.py --policy agent --maps maps_gen/heldout
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def load_index(maps_dir: Path) -> list[dict]:
    return json.loads((maps_dir / "index.json").read_text())


def run_one(args) -> dict:
    item, policy, weights, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from doom_env import make_env

    from scorer import Tracker
    from submission.actions import to_buttons

    env = make_env(item["path"], item.get("map", "MAP01"), episode_ticks=item["ticks"], render_buffers=True)
    obs, _ = env.reset(seed=seed)
    tracker = Tracker(env.privileged_state())
    grid = Path(item["path"]).with_suffix(".txt").read_text() if Path(item["path"]).with_suffix(".txt").exists() else None
    crashed = False
    if policy == "teacher":
        from teacher import Teacher, door_open_fn

        t = Teacher(grid, np.random.default_rng(seed))
        door_open = door_open_fn(env, grid)
        step = lambda o: to_buttons(t.act(o, env.privileged_state(), door_open))  # noqa: E731
    elif policy == "random":
        rng = np.random.default_rng(seed)
        step = lambda o: rng.integers(0, 2, 9)  # noqa: E731
    else:
        import torch

        torch.set_num_threads(1)
        from submission.agent import DoomAgent

        agent = DoomAgent(env.observation_space, env.action_space, "cpu") if weights is None else DoomAgent(env.observation_space, env.action_space, "cpu", weights_path=weights)
        agent.reset()
        step = agent.act
    t0 = time.perf_counter()
    while True:
        try:
            a = step(obs)
        except Exception as e:  # count as a crash: player stands still for the rest of the map
            crashed = True
            print("crash", item["path"], repr(e), file=sys.stderr)
            a = np.zeros(9, np.int8)
            step = lambda o: np.zeros(9, np.int8)  # noqa: E731
        obs, r, term, trunc, info = env.step(a)
        tracker.update(env.privileged_state())
        if term or trunc:
            break
    res = tracker.result(Path(item["path"]).name + ":" + item.get("map", "MAP01"), item["tier"], crashed).as_dict()
    res["seconds"] = time.perf_counter() - t0
    env.close()
    return res


def evaluate(maps_dir, policy="teacher", weights=None, workers=7, limit=None, seed=0) -> tuple[dict, list[dict]]:
    from scorer import EpisodeResult, summarize

    items = load_index(Path(maps_dir))
    if limit:
        items = items[:limit]
    jobs = [(it, policy, weights, seed + k) for k, it in enumerate(items)]
    with Pool(workers) as pool:
        rows = pool.map(run_one, jobs, chunksize=1)
    results = [EpisodeResult(**{k: v for k, v in r.items() if k not in ("score", "seconds")}) for r in rows]
    return summarize(results), rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default=str(ROOT / "maps_gen" / "heldout"))
    ap.add_argument("--policy", default="teacher", choices=["teacher", "random", "agent"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    summary, rows = evaluate(a.maps, a.policy, a.weights, a.workers, a.limit)
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
