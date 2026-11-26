"""Imitation learning with DAgger: the network learns to copy the scripted teacher.

Round 0: the teacher plays and we record (observation, teacher action).
Round k: the network plays (with probability 1 - beta per step), the teacher labels every visited state
with what it would have done, and the new data is added to the pool. This teaches the network to recover
from its own mistakes, which plain behaviour cloning does not.

    uv run python training/dagger.py --rounds 5 --workers 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

GOAL_ID = {"": 0, "exit": 1, "key": 2, "frontier": 3, "revisit": 4}


# --------------------------------------------------------------------------------------- collection
def collect_episode(args) -> dict:
    item, weights, beta, seed, temperature = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from doom_env import make_env

    from actions import NEXTW_ACTIONS, NUM_ACTIONS, USE_ACTIONS, to_buttons
    from model import Perception, Policy
    from model import OBS_DIM
    from scorer import Tracker
    from teacher import Teacher, dir_bin, door_open_fn

    rng = np.random.default_rng(seed)
    grid = Path(item["path"]).with_suffix(".txt").read_text()
    env = make_env(item["path"], "MAP01", episode_ticks=item["ticks"], render_buffers=False)
    obs, _ = env.reset(seed=int(seed % (2**31)))
    teacher = Teacher(grid, np.random.default_rng(seed + 1))
    door_open = door_open_fn(env, grid)
    model = None
    if beta < 1.0 and weights:
        model = Policy()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
        model.eval()
        h = model.initial_state(1)
    tracker = Tracker(env.privileged_state())
    perception = Perception()
    X, M, PA, Y, G, L, D = [], [], [], [], [], [], []
    prev, locked_seen = 0, 0
    while True:
        priv = env.privileged_state()
        label = teacher.act(obs, priv, door_open)
        x = perception.see(obs)
        locked_seen = int(locked_seen or bool(np.any(obs["rays"]["hit_type"] == 3)))
        X.append(x[:OBS_DIM].astype(np.float16)); M.append(x[OBS_DIM:].astype(np.uint8)); PA.append(prev); Y.append(label); G.append(GOAL_ID.get(teacher.goal_kind, 0)); L.append(locked_seen); D.append(dir_bin(teacher.dir_rel))
        a = label
        if model is not None:
            logits, _, h = model.step(torch.from_numpy(x).unsqueeze(0), torch.tensor([prev]), h)
            if rng.random() >= beta:
                lg = logits[0].double().numpy()
                if prev in USE_ACTIONS:
                    lg[list(USE_ACTIONS)] = -np.inf
                if prev in NEXTW_ACTIONS:
                    lg[list(NEXTW_ACTIONS)] = -np.inf
                z = (lg - lg.max()) / temperature
                p = np.exp(z); p /= p.sum()
                a = int(rng.choice(NUM_ACTIONS, p=p))
        # keep the teacher's own memory of its last action consistent with what was executed
        teacher.prev_action = a
        teacher.prev_use = a in USE_ACTIONS
        buttons = to_buttons(a)
        perception.did(buttons)
        obs, r, term, trunc, info = env.step(buttons)
        tracker.update(env.privileged_state())
        prev = a
        if term or trunc:
            break
    env.close()
    res = tracker.result(Path(item["path"]).name, item["tier"]).as_dict()
    return {"x": np.stack(X), "m": np.stack(M), "pa": np.array(PA, np.int16), "y": np.array(Y, np.int16), "g": np.array(G, np.int8),
            "l": np.array(L, np.int8), "d": np.array(D, np.int8), "result": res}


def collect(items, weights, beta, workers, seed0, temperature=0.5):
    jobs = [(it, weights, beta, seed0 + k, temperature) for k, it in enumerate(items)]
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        return pool.map(collect_episode, jobs, chunksize=1)


# --------------------------------------------------------------------------------------- training
def train(model, episodes, epochs, device, lr=3e-4, batch=64, chunk=64, log=print):
    import torch
    import torch.nn.functional as Fn

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        order = np.random.permutation(len(episodes))
        tot, n, correct, dcorrect = 0.0, 0, 0, 0.0
        t0 = time.time()
        for b0 in range(0, len(order), batch):
            eps = [episodes[i] for i in order[b0:b0 + batch]]
            T = max(len(e["y"]) for e in eps)
            B = len(eps)
            x = np.zeros((B, T, eps[0]["x"].shape[1] + eps[0]["m"].shape[1]), np.float32)
            pa = np.zeros((B, T), np.int64)
            y = np.zeros((B, T), np.int64)
            g = np.zeros((B, T), np.int64)
            dd = np.zeros((B, T), np.int64)
            lk = np.zeros((B, T), np.float32)
            m = np.zeros((B, T), np.float32)
            for i, e in enumerate(eps):
                k = len(e["y"])
                x[i, :k, :e["x"].shape[1]] = e["x"]; x[i, :k, e["x"].shape[1]:] = e["m"]; pa[i, :k] = e["pa"]; y[i, :k] = e["y"]; g[i, :k] = e["g"]; dd[i, :k] = e["d"]; lk[i, :k] = e["l"]; m[i, :k] = 1
            x, pa, y, g, dd, lk, m = (torch.from_numpy(a).to(device) for a in (x, pa, y, g, dd, lk, m))
            h = model.initial_state(B, device)
            for t0c in range(0, T, chunk):
                sl = slice(t0c, t0c + chunk)
                mm = m[:, sl]
                if mm.sum() == 0:
                    break
                logits, _, (ag, al, ad), h = model(x[:, sl], pa[:, sl], h)
                h = h.detach()
                ce = Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, sl].reshape(-1), reduction="none", label_smoothing=0.05)
                ceg = Fn.cross_entropy(ag.reshape(-1, ag.shape[-1]), g[:, sl].reshape(-1), reduction="none")
                bce = Fn.binary_cross_entropy_with_logits(al.reshape(-1), lk[:, sl].reshape(-1), reduction="none")
                mf = mm.reshape(-1)
                ced = Fn.cross_entropy(ad.reshape(-1, ad.shape[-1]), dd[:, sl].reshape(-1), reduction="none")
                loss = ((ce + 0.2 * ceg + 0.2 * bce + 0.5 * ced) * mf).sum() / mf.sum()
                dcorrect += float(((ad.argmax(-1) == dd[:, sl]).float() * mm).sum())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * float(mf.sum()); n += float(mf.sum())
                correct += float(((logits.argmax(-1) == y[:, sl]).float() * mm).sum())
        log(f"  epoch {ep}: loss {tot / n:.3f}  acc {correct / n:.3f}  dir-acc {dcorrect / n:.3f}  ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default=str(ROOT / "maps_gen" / "train"))
    ap.add_argument("--heldout", nargs="+", default=[str(ROOT / "maps_gen" / "heldout"), str(ROOT / "maps_gen" / "heldout2")],
                    help="held-out map sets; the score is the map-count-weighted mean over them")
    ap.add_argument("--out", default=str(ROOT / "runs" / "dagger"))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--episodes", type=int, default=400, help="episodes collected per round")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-pool", type=int, default=2500, help="max episodes kept in the aggregated dataset")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume-beta", type=float, default=0.25, help="teacher mix per step when resuming")
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    import torch

    torch.set_num_threads(8)
    from evaluate import evaluate
    from model import Policy

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    items = json.loads((Path(a.maps) / "index.json").read_text())
    model = Policy().to(a.device)
    if a.resume:
        from model import load_compatible

        log(f"resume from {a.resume}; adapted layers: {load_compatible(model, torch.load(a.resume, map_location=a.device, weights_only=True)['model'])}")
    if a.resume:  # so round 0 already lets the resumed network play (not just the teacher)
        torch.save({"model": model.state_dict()}, out / "last.pt")
    pool: list[dict] = []
    best = -1e9
    betas = [1.0, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2]
    rng = np.random.default_rng(0)
    for rnd in range(a.rounds):
        beta = betas[min(rnd, len(betas) - 1)] if not a.resume else a.resume_beta
        weights = str(out / "last.pt") if (out / "last.pt").exists() else None
        chosen = [items[i] for i in rng.choice(len(items), size=min(a.episodes, len(items)), replace=False)]
        t0 = time.time()
        eps = collect(chosen, weights, beta, a.workers, seed0=rnd * 100000 + 17)
        steps = sum(len(e["y"]) for e in eps)
        exits = np.mean([e["result"]["exit"] for e in eps])
        log(f"round {rnd} beta {beta}: collected {len(eps)} episodes, {steps} steps in {time.time() - t0:.0f}s, behaviour exit rate {exits:.2f}")
        for e in eps:
            e.pop("result")
        pool += eps
        if len(pool) > a.max_pool:
            pool = pool[-a.max_pool:]
        train(model, pool, a.epochs, a.device, lr=a.lr, log=log)
        torch.save({"model": model.state_dict()}, out / "last.pt")
        parts = [(evaluate(h, "agent", str(out / "last.pt"), a.workers)[0]) for h in a.heldout]
        n_tot = sum(p_["maps"] for p_ in parts)
        summary = {"weighted": sum(p_["weighted"] * p_["maps"] for p_ in parts) / n_tot}
        for k in ("t1", "t2", "t3", "t4"):
            if all(k in p_ for p_ in parts):
                summary[k] = {f: sum(p_[k][f] * p_["maps"] for p_ in parts) / n_tot for f in ("score", "exit", "dead", "stuck")}
        log(f"round {rnd} held-out ({n_tot} maps): weighted {summary['weighted']:.1f} " + " ".join(
            f"t{k}: score {summary[f't{k}']['score']:.1f} exit {summary[f't{k}']['exit']:.2f} dead {summary[f't{k}']['dead']:.2f} stuck {summary[f't{k}']['stuck']:.2f}"
            for k in (1, 2, 3, 4) if f"t{k}" in summary))
        if summary["weighted"] > best:
            best = summary["weighted"]
            torch.save({"model": model.state_dict(), "round": rnd, "heldout": summary}, out / "best.pt")
            log(f"  new best {best:.1f}")


if __name__ == "__main__":
    main()
