"""Our own recurrent PPO fine-tune, starting from the imitation (DAgger) policy.

- Rewards mirror the local scorer (exit, keys, doors, locks, new areas, kills, minus damage and death),
  computed from privileged state during training only.
- A fading behaviour-cloning term (cross-entropy to the teacher's label for every visited state) keeps the
  policy close to what it learned by imitation while RL improves it.
- Full-episode rollouts, GAE, clipped policy loss, value loss, entropy bonus; truncated BPTT over the GRU
  with the hidden state carried across chunks.
- A checkpoint is kept only if it beats the best one so far on the held-out maps.

    uv run python training/ppo.py --init runs/dagger1/best.pt --iters 200 --device mps
"""

from __future__ import annotations

import argparse
import json
import math
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

REWARD = {"exit": 10.0, "key": 3.0, "door": 0.5, "lock": 3.0, "area": 0.15, "kill": 0.5, "damage": 0.01, "death": 5.0, "step": 0.002}


def rollout(args) -> dict:
    item, weights, seed, temperature = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from doom_env import make_env

    from actions import NEXTW_ACTIONS, NUM_ACTIONS, USE_ACTIONS, to_buttons
    from model import Policy, encode_obs
    from scorer import AREA, Tracker
    from teacher import Teacher, door_open_fn

    rng = np.random.default_rng(seed)
    grid = Path(item["path"]).with_suffix(".txt").read_text()
    env = make_env(item["path"], "MAP01", episode_ticks=item["ticks"], render_buffers=False)
    obs, _ = env.reset(seed=int(seed % (2**31)))
    teacher = Teacher(grid, np.random.default_rng(seed + 1))
    door_open = door_open_fn(env, grid)
    model = Policy()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"])
    model.eval()
    h = model.initial_state(1)
    p0 = env.privileged_state()
    tracker = Tracker(p0)
    areas = {(math.floor(p0["x"] / AREA), math.floor(p0["y"] / AREA))}
    last = {"keys": 0, "doors": 0, "locks": 0, "kills": 0, "damage": 0.0}
    X, PA, A, LP, R, Y = [], [], [], [], [], []
    prev = 0
    while True:
        label = teacher.act(obs, env.privileged_state(), door_open)
        x = encode_obs(obs)
        logits, _, h = model.step(torch.from_numpy(x).unsqueeze(0), torch.tensor([prev]), h)
        lg = logits[0].double().numpy()
        if prev in USE_ACTIONS:
            lg[list(USE_ACTIONS)] = -np.inf
        if prev in NEXTW_ACTIONS:
            lg[list(NEXTW_ACTIONS)] = -np.inf
        z = (lg - lg.max()) / temperature
        p = np.exp(z); p /= p.sum()
        a = int(rng.choice(NUM_ACTIONS, p=p))
        X.append(x.astype(np.float16)); PA.append(prev); A.append(a); LP.append(math.log(max(p[a], 1e-12))); Y.append(label)
        teacher.prev_action, teacher.prev_use = a, a in USE_ACTIONS
        obs, _, term, trunc, info = env.step(to_buttons(a))
        pr = env.privileged_state()
        tracker.update(pr)
        res = tracker.result("", item["tier"])
        r = -REWARD["step"]
        cell = (math.floor(pr["x"] / AREA), math.floor(pr["y"] / AREA))
        if cell not in areas:
            areas.add(cell); r += REWARD["area"]
        r += REWARD["key"] * (res.keys - last["keys"]) + REWARD["door"] * (res.doors_opened - last["doors"])
        r += REWARD["lock"] * (res.locks_opened - last["locks"]) + REWARD["kill"] * (res.kills - last["kills"])
        r -= REWARD["damage"] * (res.damage - last["damage"])
        last = {"keys": res.keys, "doors": res.doors_opened, "locks": res.locks_opened, "kills": res.kills, "damage": res.damage}
        if res.exit:
            r += REWARD["exit"] * (1.0 + 0.5 * (1.0 - res.tic / res.ticks))
        if res.dead:
            r -= REWARD["death"]
        R.append(r)
        prev = a
        if term or trunc:
            break
    env.close()
    out = tracker.result(Path(item["path"]).name, item["tier"]).as_dict()
    return {"x": np.stack(X), "pa": np.array(PA, np.int16), "a": np.array(A, np.int16), "lp": np.array(LP, np.float32),
            "r": np.array(R, np.float32), "y": np.array(Y, np.int16), "truncated": bool(trunc and not term), "result": out}


def masked_logits(logits, prev, use_ids, nextw_ids):
    import torch

    mask = torch.zeros_like(logits, dtype=torch.bool)
    pu = torch.isin(prev, use_ids).unsqueeze(-1) & torch.isin(torch.arange(logits.shape[-1], device=logits.device), use_ids)
    pn = torch.isin(prev, nextw_ids).unsqueeze(-1) & torch.isin(torch.arange(logits.shape[-1], device=logits.device), nextw_ids)
    mask = pu | pn
    return logits.masked_fill(mask, -1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--maps", default=str(ROOT / "maps_gen" / "train"))
    ap.add_argument("--heldout", default=str(ROOT / "maps_gen" / "heldout"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "ppo1"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=28)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--bc", type=float, default=0.5, help="initial weight of the teacher cross-entropy term")
    ap.add_argument("--bc-decay", type=float, default=0.985)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as Fn

    from actions import NEXTW_ACTIONS, USE_ACTIONS
    from evaluate import evaluate
    from model import Policy

    torch.set_num_threads(4)
    dev = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    model = Policy().to(dev)
    model.load_state_dict(torch.load(a.init, map_location=dev, weights_only=True)["model"])
    torch.save({"model": model.state_dict()}, out / "last.pt")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    items = json.loads((Path(a.maps) / "index.json").read_text())
    use_ids = torch.tensor(USE_ACTIONS, device=dev)
    nextw_ids = torch.tensor(NEXTW_ACTIONS, device=dev)

    summary, _ = evaluate(a.heldout, "agent", a.init, a.workers)
    best = summary["weighted"]
    log(f"start: held-out weighted {best:.1f}")
    torch.save({"model": model.state_dict(), "iter": 0, "heldout": summary}, out / "best.pt")

    rng = np.random.default_rng(1)
    bc = a.bc
    ctx = get_context("spawn")
    pool = ctx.Pool(a.workers)
    ret_rms = 1.0
    for it in range(1, a.iters + 1):
        t0 = time.time()
        chosen = [items[i] for i in rng.choice(len(items), size=a.episodes, replace=False)]
        eps = pool.map(rollout, [(c, str(out / "last.pt"), int(rng.integers(1 << 30)), a.temperature) for c in chosen], chunksize=1)
        t_col = time.time() - t0

        # ---- batch episodes, compute values and GAE
        B = len(eps)
        T = max(len(e["a"]) for e in eps)
        D = eps[0]["x"].shape[1]
        x = np.zeros((B, T, D), np.float32); pa = np.zeros((B, T), np.int64); act = np.zeros((B, T), np.int64)
        lp_old = np.zeros((B, T), np.float32); rew = np.zeros((B, T), np.float32); y = np.zeros((B, T), np.int64); m = np.zeros((B, T), np.float32)
        for i, e in enumerate(eps):
            k = len(e["a"])
            x[i, :k] = e["x"]; pa[i, :k] = e["pa"]; act[i, :k] = e["a"]; lp_old[i, :k] = e["lp"]; rew[i, :k] = e["r"]; y[i, :k] = e["y"]; m[i, :k] = 1
        tx, tpa, tact, tlp, ty, tm = (torch.from_numpy(v).to(dev) for v in (x, pa, act, lp_old, y, m))
        with torch.no_grad():
            _, values, _, _ = model(tx, tpa, model.initial_state(B, dev))
        values = values.float().cpu().numpy() * m
        # reward scaling by a running estimate of return std
        ret_rms = 0.95 * ret_rms + 0.05 * float(np.std(rew[m > 0]) * math.sqrt(1.0 / (1 - a.gamma)) + 1e-3)
        scale = 1.0 / max(ret_rms, 1e-3)
        adv = np.zeros_like(rew)
        for i, e in enumerate(eps):
            k = len(e["a"])
            last_gae = 0.0
            boot = values[i, k - 1] if e["truncated"] else 0.0
            for t in range(k - 1, -1, -1):
                nv = boot if t == k - 1 else values[i, t + 1]
                delta = rew[i, t] * scale + a.gamma * nv - values[i, t]
                last_gae = delta + a.gamma * a.lam * last_gae
                adv[i, t] = last_gae
        ret = adv + values
        am = adv[m > 0]
        adv = (adv - am.mean()) / (am.std() + 1e-8) * m
        tadv, tret = torch.from_numpy(adv).to(dev), torch.from_numpy(ret.astype(np.float32)).to(dev)

        # ---- PPO update (truncated BPTT, chunks of 64, hidden carried)
        model.train()
        stats = []
        for _ in range(a.epochs):
            for idx in np.array_split(rng.permutation(B), max(1, B // 14)):
                idx_t = torch.from_numpy(idx).to(dev)
                h = model.initial_state(len(idx), dev)
                for c0 in range(0, T, 64):
                    sl = slice(c0, c0 + 64)
                    mm = tm[idx_t, sl]
                    if mm.sum() == 0:
                        break
                    logits, v, _, h = model(tx[idx_t, sl], tpa[idx_t, sl], h)
                    h = h.detach()
                    logits = masked_logits(logits, tpa[idx_t, sl], use_ids, nextw_ids) / a.temperature
                    logp_all = torch.log_softmax(logits, -1)
                    lp = logp_all.gather(-1, tact[idx_t, sl].unsqueeze(-1)).squeeze(-1)
                    ratio = torch.exp(lp - tlp[idx_t, sl])
                    ad = tadv[idx_t, sl]
                    pl = -torch.min(ratio * ad, ratio.clamp(1 - a.clip, 1 + a.clip) * ad)
                    vl = (v - tret[idx_t, sl]) ** 2
                    ent = -(logp_all.exp() * logp_all).sum(-1)
                    bcl = Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]), ty[idx_t, sl].reshape(-1), reduction="none").view_as(lp)
                    loss = ((pl + 0.5 * vl - a.ent * ent + bc * bcl) * mm).sum() / mm.sum()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    stats.append([float((pl * mm).sum() / mm.sum()), float((vl * mm).sum() / mm.sum()), float((ent * mm).sum() / mm.sum()),
                                  float((bcl * mm).sum() / mm.sum()), float((((ratio - 1).abs() > a.clip).float() * mm).sum() / mm.sum())])
        model.eval()
        torch.save({"model": model.state_dict()}, out / "last.pt")
        bc *= a.bc_decay
        s = np.mean(stats, 0)
        res = [e["result"] for e in eps]
        log(f"it {it}: exit {np.mean([r['exit'] for r in res]):.2f} dead {np.mean([r['dead'] for r in res]):.2f} "
            f"score {np.mean([r['score'] for r in res]):.1f} | pl {s[0]:.3f} vl {s[1]:.3f} ent {s[2]:.2f} bc {s[3]:.2f} clip {s[4]:.2f} "
            f"bcw {bc:.3f} | {sum(len(e['a']) for e in eps)} steps, collect {t_col:.0f}s total {time.time() - t0:.0f}s")
        if it % a.eval_every == 0:
            summary, _ = evaluate(a.heldout, "agent", str(out / "last.pt"), a.workers)
            log(f"it {it} held-out: weighted {summary['weighted']:.1f} " + " ".join(
                f"t{k}: {summary[f't{k}']['score']:.1f}/{summary[f't{k}']['exit']:.2f}" for k in (1, 2, 3, 4) if f"t{k}" in summary))
            if summary["weighted"] > best:
                best = summary["weighted"]
                torch.save({"model": model.state_dict(), "iter": it, "heldout": summary}, out / "best.pt")
                log(f"  new best {best:.1f}")
    pool.close()


if __name__ == "__main__":
    main()
