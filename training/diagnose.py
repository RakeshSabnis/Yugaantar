"""Failure analysis: where do runs on a tier break down?

For every map of the tier it records the timeline (step of each key pickup / lock opening), how the run
ended, and how the time was spent (near a locked door without / with the right key, stuck, fighting).

    uv run python training/diagnose.py --tier 4 --policy agent
    uv run python training/diagnose.py --tier 4 --policy teacher
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_OF = {"R": "r", "B": "b", "Y": "y"}
VITAL = {"b": 8, "y": 9, "r": 10}


def run(args):
    item, policy, weights, seed = args
    os.environ["OMP_NUM_THREADS"] = "1"
    for p in (ROOT / "training", ROOT / "submission", ROOT):
        sys.path.insert(0, str(p))
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from doom_env import make_env

    from actions import to_buttons
    from mapgen import CELL
    from scorer import Tracker
    from teacher import Teacher, door_open_fn

    grid = Path(item["path"]).with_suffix(".txt").read_text()
    rows = grid.strip("\n").split("\n")
    h = len(rows)
    locks = [(r, c, ch) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in KEY_OF]
    env = make_env(item["path"], "MAP01", episode_ticks=item["ticks"], render_buffers=True)  # frames on: the door-colour filter reads them
    obs, _ = env.reset(seed=seed)
    door_open = door_open_fn(env, grid)
    if policy == "teacher":
        t = Teacher(grid, np.random.default_rng(seed))
        step = lambda o: to_buttons(t.act(o, env.privileged_state(), door_open))  # noqa: E731
    else:
        from agent import DoomAgent

        ag = DoomAgent(env.observation_space, env.action_space, "cpu", weights_path=weights)
        ag.reset()
        step = ag.act
    tr = Tracker(env.privileged_state())
    ev = {}
    n = at_lock_nokey = at_lock_key = fighting = 0
    held_prev: set = set()
    opened_prev = 0
    while True:
        a = step(obs)
        obs, _, term, trunc, _ = env.step(a)
        pr = env.privileged_state()
        tr.update(pr)
        n += 1
        held = {k for k, i in VITAL.items() if obs["vitals"][i] > 0}
        for k in held - held_prev:
            ev[f"key_{len(held)}"] = n
        held_prev = held
        res = tr.result("", 4)
        if res.locks_opened > opened_prev:
            ev[f"lock_{res.locks_opened}"] = n
            opened_prev = res.locks_opened
        ents = obs["visible_entities"]
        if np.any((np.asarray(ents["mask"]) > 0) & np.isin(np.asarray(ents["entity_type"]), (1, 2, 3))):
            fighting += 1
        for r, c, ch in locks:
            if door_open(r, c):
                continue
            x, y = c * CELL + CELL / 2, (h - 1 - r) * CELL + CELL / 2
            if math.hypot(pr["x"] - x, pr["y"] - y) < 110:
                if KEY_OF[ch] in held:
                    at_lock_key += 1
                else:
                    at_lock_nokey += 1
                break
        if term or trunc:
            break
    env.close()
    res = tr.result(Path(item["path"]).name, 4)
    if res.exit:
        end = "exit"
    elif res.dead:
        end = "died"
    else:
        end = "timeout"
    stage = f"keys{res.keys}_locks{res.locks_opened}"
    return {"map": res.map, "end": end, "stage": stage, "steps": n, "cap": item["ticks"] // 4, "events": ev,
            "at_lock_nokey": at_lock_nokey / n, "at_lock_key": at_lock_key / n, "stuck": res.stuck, "fighting": fighting / n,
            "areas": res.areas, "kills": res.kills, "health": res.health, "n_locks": len({ch for _, _, ch in locks})}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=4)
    ap.add_argument("--policy", default="agent")
    ap.add_argument("--weights", default=str(ROOT / "submission" / "weights" / "policy.pt"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    items = [it for m in ("heldout", "heldout2") for it in json.loads((ROOT / "maps_gen" / m / "index.json").read_text()) if it["tier"] == a.tier]
    with get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(run, [(it, a.policy, a.weights, 1000 + k) for k, it in enumerate(items)], chunksize=1)
    if a.out:
        Path(a.out).write_text(json.dumps(rows))
    n = len(rows)
    print(f"{a.policy}: {n} tier-{a.tier} maps")
    print(" how runs end:", dict(Counter(r["end"] for r in rows)))
    print(" furthest stage reached (non-exit runs):", dict(sorted(Counter(r["stage"] for r in rows if r["end"] != "exit").items())))
    print(" stage at death:", dict(sorted(Counter(r["stage"] for r in rows if r["end"] == "died").items())))
    print(" stage at timeout:", dict(sorted(Counter(r["stage"] for r in rows if r["end"] == "timeout").items())))
    for k in ("key_1", "lock_1", "key_2", "lock_2"):
        v = [r["events"][k] for r in rows if k in r["events"]]
        print(f" {k}: reached in {len(v)}/{n} runs" + (f", median step {sorted(v)[len(v) // 2]}" if v else ""))
    ex = [r["steps"] for r in rows if r["end"] == "exit"]
    print(" exit: median step", sorted(ex)[len(ex) // 2] if ex else None, "| step cap", rows[0]["cap"])
    to = [r for r in rows if r["end"] == "timeout"]
    if to:
        f = lambda k: sum(r[k] for r in to) / len(to)  # noqa: E731
        print(f" time use in timeout runs: at locked door WITHOUT its key {100 * f('at_lock_nokey'):.0f}%, at locked door WITH key {100 * f('at_lock_key'):.0f}%, "
              f"stuck {100 * f('stuck'):.0f}%, enemy in view {100 * f('fighting'):.0f}%, areas {f('areas'):.0f}")
