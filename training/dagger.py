"""Imitation stage: DAgger from the scripted teacher to the recurrent policy.

Each iteration (1) plays generated levels, with the teacher driving at first and the current network
driving more and more later, while the teacher labels every step with what it would have done;
(2) retrains the network on everything collected so far; (3) scores the network alone on held-out
levels. Letting the network drive is what teaches it to recover from its own mistakes, and the
inputs are always the network's own (dead-reckoned) memory, exactly as at evaluation.

    python -m training.dagger --workers 14 --iters 8 --episodes 400 --train-steps 3000   # AWS
    python -m training.dagger --smoke                                                     # laptop sanity check
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
KEYS = ("rays", "ent", "ent_mask", "body", "map")


# ----------------------------------------------------------------------------- collection
_AGENT = None


def _collect(job) -> dict:
    seed, beta, out_dir = job
    global _AGENT
    import tempfile

    from doom_env import make_env
    from doom_env.spaces import action_space, observation_space
    from training.mapgen import make_level, write_level
    from training.scorer import EpisodeScore
    from training.teacher import Teacher

    if _AGENT is None:
        from submission.agent import DoomAgent
        _AGENT = DoomAgent(observation_space(), action_space(), "cpu")
    agent = _AGENT
    rng = np.random.default_rng([seed, 77])
    lvl = make_level(seed)
    wad = Path(tempfile.gettempdir()) / f"dagger_{os.getpid()}.wad"
    write_level(lvl, wad)
    env = make_env(wad, episode_ticks=lvl["ticks"])
    buf = {k: [] for k in KEYS + ("label",)}
    try:
        obs, _ = env.reset(seed=seed)
        teacher = Teacher(rng)
        teacher.reset(env)
        agent.reset()
        agent.rng = rng
        score = EpisodeScore(lvl["monsters"], max(1, lvl["floor_cells"] * lvl["cell"] ** 2 // 128**2))
        score.update(env.privileged_state())
        student = False
        t = 0
        while True:
            if t % 40 == 0:  # who drives is re-drawn every 40 steps, so states mix both behaviours
                student = rng.random() >= beta
            p = agent.probs(obs)
            label = teacher.act(obs, env)
            x = agent.last_x
            for k in KEYS:
                buf[k].append(np.packbits(x[k].astype(bool)) if k == "map" else x[k])
            buf["label"].append(label)
            a = agent.mask((rng.random(9) < p).astype(np.int8), p) if student else label.copy()
            agent.prev, agent.t = a, agent.t + 1
            obs, _, term, trunc, _ = env.step(a)
            score.update(env.privileged_state())
            t += 1
            if term or trunc:
                break
        # dead-reckoning error of the network's memory at the last step (map units)
        tm, sm = teacher.map, agent.mapper
        drift = float(math.hypot(tm.x - sm.x, tm.y - sm.y))
    finally:
        env.close()
    np.savez_compressed(
        Path(out_dir) / f"ep_{seed:07d}.npz",
        rays=np.asarray(buf["rays"], np.float16), ent=np.asarray(buf["ent"], np.float16), ent_mask=np.asarray(buf["ent_mask"], bool),
        body=np.asarray(buf["body"], np.float16), map=np.asarray(buf["map"], np.uint8), label=np.asarray(buf["label"], np.uint8))
    return {**score.result(), "tier": lvl["tier"], "seed": seed, "steps": t, "drift": drift}


def _collect_safe(job):
    """One broken level must not take down a long run: log it and carry on."""
    try:
        return _collect(job)
    except Exception as e:  # noqa: BLE001
        print(f"[collect] seed {job[0]} failed: {type(e).__name__}: {e}", flush=True)
        return None


def _eval_safe(job):
    from training.play import _work
    try:
        return _work(job)
    except Exception as e:  # noqa: BLE001
        print(f"[eval] seed {job[1]} failed: {type(e).__name__}: {e}", flush=True)
        return None


def _init_worker() -> None:
    from training.play import _low_priority
    _low_priority()
    import torch
    torch.set_num_threads(1)


def collect(seeds, beta: float, out_dir: Path, workers: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, beta, str(out_dir)) for s in seeds]
    with mp.Pool(workers, initializer=_init_worker) as pool:
        return [r for r in pool.map(_collect_safe, jobs, chunksize=1) if r is not None]


# ----------------------------------------------------------------------------- training
def load(dirs: list[Path]) -> list[dict]:
    eps = []
    for d in dirs:
        for f in sorted(d.glob("ep_*.npz")):
            with np.load(f) as z:
                eps.append({k: z[k] for k in z.files})
    return eps


def train(eps: list[dict], steps: int, device: str, init: Path | None, out: Path, batch: int = 32, seq: int = 96,
          burn: int = 16, lr: float = 3e-4, log=print) -> None:
    import torch
    import torch.nn.functional as F

    from model import Policy

    net = Policy().to(device)
    if init is not None and init.is_file():
        net.load_state_dict(torch.load(init, map_location=device, weights_only=True)["model"])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    lengths = np.array([len(e["label"]) for e in eps])
    freq = np.clip(sum(e["label"].sum(0) for e in eps) / lengths.sum(), 1e-3, 1 - 1e-3)
    # rare buttons (USE fires once per door) would otherwise be drowned out
    pos_w = torch.tensor(np.clip(np.sqrt((1 - freq) / freq), 0.5, 6.0), dtype=torch.float32, device=device)
    rng = np.random.default_rng(0)
    probs = lengths / lengths.sum()
    net.train()
    t0, run = time.time(), 0.0
    for step in range(1, steps + 1):
        xb = {k: [] for k in KEYS}
        yb, wb = [], []
        for ei in rng.choice(len(eps), size=batch, p=probs):
            e = eps[ei]
            n = len(e["label"])
            s = 0 if n <= seq or rng.random() < 0.15 else int(rng.integers(0, n - seq + 1))
            sl = slice(s, s + seq)
            pad = seq - min(seq, n - s)
            for k in KEYS:
                v = e[k][sl]
                if k == "map":
                    v = np.unpackbits(v, axis=1)[:, : 14 * 32 * 32].reshape(-1, 14, 32, 32)
                if pad:
                    v = np.concatenate([v, np.zeros((pad,) + v.shape[1:], v.dtype)])
                xb[k].append(v)
            y = e["label"][sl]
            w = np.ones(len(y), np.float32)
            if s > 0:
                w[:burn] = 0.0  # the GRU starts cold mid-episode: let it warm up before it is graded
            if pad:
                y = np.concatenate([y, np.zeros((pad, 9), y.dtype)])
                w = np.concatenate([w, np.zeros(pad, np.float32)])
            yb.append(y)
            wb.append(w)
        x = {k: torch.from_numpy(np.stack(v)).to(device) for k, v in xb.items()}
        x = {k: (v if k == "ent_mask" else v.float()) for k, v in x.items()}
        y = torch.from_numpy(np.stack(yb)).to(device).float()
        w = torch.from_numpy(np.stack(wb)).to(device)
        logits, _, _ = net(x)
        loss = (F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w, reduction="none").mean(-1) * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % max(1, steps // 10) == 0 or step == steps:
            acc = (((logits > 0) == (y > 0.5)).float().mean(-1) * w).sum() / w.sum()
            log(f"  train step {step}/{steps}  loss {run:.4f}  button acc {acc.item():.3f}  {step / (time.time() - t0):.1f} it/s")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()}}, out)


# ----------------------------------------------------------------------------- loop
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=400, help="new training levels per iteration")
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--eval-levels", type=int, default=48)
    ap.add_argument("--data", default="data/dagger")
    ap.add_argument("--start-iter", type=int, default=0, help="resume: reuse data of earlier iterations")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline end to end")
    args = ap.parse_args()
    if args.smoke:
        args.workers, args.iters, args.episodes, args.train_steps, args.eval_levels, args.data = 2, 2, 4, 30, 4, "data/dagger_smoke"
    import torch

    from training.play import HELD_OUT, _low_priority, summarize

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    data = ROOT / args.data
    weights = ROOT / "submission" / "weights" / "policy.pt"
    (ROOT / "runs").mkdir(exist_ok=True)
    logf = open(ROOT / "runs" / "dagger.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"=== dagger start: device {device}, workers {args.workers}, {args.iters} iters x {args.episodes} levels")
    for it in range(args.start_iter, args.iters):
        beta = 1.0 if it == 0 else max(0.15, 0.6**it)  # share of the time the teacher drives
        seeds = range(it * args.episodes, (it + 1) * args.episodes)
        t0 = time.time()
        rows = collect(seeds, beta, data / f"iter_{it:02d}", args.workers)
        n = sum(r["steps"] for r in rows)
        log(f"iter {it}: collected {n} steps from {len(rows)} levels in {time.time() - t0:.0f}s (teacher drives {beta:.0%}); "
            f"exit {np.mean([r['exit'] for r in rows]):.2f}, memory drift median {np.median([r['drift'] for r in rows]):.0f} units")
        eps = load([data / f"iter_{k:02d}" for k in range(it + 1)])
        train(eps, args.train_steps, device, weights if it > 0 else None, weights, log=log)
        del eps
        with mp.Pool(args.workers, initializer=_low_priority) as pool:
            ev = [r for r in pool.map(_eval_safe, [("agent", s) for s in range(HELD_OUT, HELD_OUT + args.eval_levels)], chunksize=1) if r is not None]
        log(f"iter {it}: network alone on held-out levels\n{summarize(ev)}")
        (ROOT / "runs" / f"eval_iter_{it:02d}.json").write_text(json.dumps(ev))
        import shutil
        shutil.copy(weights, ROOT / "runs" / f"policy_iter_{it:02d}.pt")  # keep every round: the last is not always the best
    log("=== done; weights at submission/weights/policy.pt")


if __name__ == "__main__":
    main()
