"""Score any repo's submission/agent.py on our held-out maps with our local scorer (same maps, same seeds).

    uv run python training/compare.py --repo . --maps maps_gen/heldout maps_gen/heldout2
    uv run python training/compare.py --repo /path/to/other/checkout --maps maps_gen/heldout
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_one(args):
    repo, item, seed = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / "training"))
    from doom_env import make_env
    from scorer import Tracker

    for p in (str(Path(repo) / "submission"), str(repo)):
        sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("submission_agent", Path(repo) / "submission" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_agent"] = mod
    spec.loader.exec_module(mod)
    env = make_env(item["path"], item.get("map", "MAP01"), episode_ticks=item["ticks"], render_buffers=False)
    obs, _ = env.reset(seed=seed)
    agent = mod.DoomAgent(env.observation_space, env.action_space, "cpu")
    agent.reset()
    tr = Tracker(env.privileged_state())
    crashed, spent, n = False, 0.0, 0
    while True:
        t0 = time.perf_counter()
        try:
            a = np.asarray(agent.act(obs))
            assert a.shape == (9,) and set(np.unique(a)) <= {0, 1}
        except Exception:
            crashed = True
            a = np.zeros(9, np.int8)
        spent += time.perf_counter() - t0
        n += 1
        obs, r, term, trunc, _ = env.step(a)
        tr.update(env.privileged_state())
        if term or trunc:
            break
    env.close()
    d = tr.result(Path(item["path"]).name, item["tier"], crashed).as_dict()
    d["ms"] = spent / n * 1000
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--maps", nargs="+", default=[str(ROOT / "maps_gen" / "heldout")])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "training"))
    from scorer import EpisodeResult, summarize

    items = [it for m in a.maps for it in json.loads((Path(m) / "index.json").read_text())]
    with get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(run_one, [(str(Path(a.repo).resolve()), it, 1000 + k) for k, it in enumerate(items)], chunksize=1)
    s = summarize([EpisodeResult(**{k: v for k, v in r.items() if k not in ("score", "ms")}) for r in rows])
    s["ms_per_decision"] = sum(r["ms"] for r in rows) / len(rows)
    print(json.dumps(s))
