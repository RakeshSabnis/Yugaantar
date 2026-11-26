"""Run a policy on generated levels and score it with the local scorer.

    python -m training.play --policy teacher --seeds 1000000:1000040      # held-out seeds
    python -m training.play --policy agent --seeds 1000000:1000040        # submission/agent.py
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from doom_env import make_env
from training.mapgen import make_level, write_level
from training.scorer import TIER_WEIGHT, EpisodeScore

HELD_OUT = 1_000_000  # seeds >= this are never used for training
REAL_EVERY = 5  # every 5th training seed is a hand-made Freedoom level instead of a generated one


def real_level(seed: int) -> dict:
    """A hand-made Freedoom 2 level (MAP01-MAP32, bundled with ViZDoom): diagonal walls, stairs,
    odd room shapes. Nothing like our grid generator, which is the point: the agent must not depend
    on one map style. Reported as "tier 5" and not counted in the tier-weighted score."""
    import vizdoom
    return {"wad": os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad"), "map": f"MAP{seed // REAL_EVERY % 32 + 1:02d}",
            "ticks": 4200, "tier": 5, "seed": seed, "monsters": None}


def level_for(seed: int, real: bool = True) -> dict:
    return real_level(seed) if real and seed % REAL_EVERY == REAL_EVERY - 1 else make_level(seed)


def run_episode(lvl: dict, policy, wad_dir: str | None = None, on_step=None) -> dict:
    """policy: object with reset(env) and act(obs, env) -> action."""
    if "wad" in lvl:
        env = make_env(lvl["wad"], lvl["map"], episode_ticks=lvl["ticks"])
    else:
        wad = Path(wad_dir or tempfile.gettempdir()) / f"lvl_{os.getpid()}.wad"
        write_level(lvl, wad)
        env = make_env(wad, episode_ticks=lvl["ticks"])  # frames are needed for the exit cue
    try:
        obs, _ = env.reset(seed=lvl["seed"])
        score = EpisodeScore() if "wad" in lvl else EpisodeScore(lvl["monsters"], max(1, lvl["floor_cells"] * lvl["cell"] ** 2 // 128 ** 2))
        score.update(env.privileged_state())
        policy.reset(env)
        while True:
            action = policy.act(obs, env)
            if on_step is not None:
                on_step(obs, action, env)
            obs, _, term, trunc, _ = env.step(action)
            score.update(env.privileged_state())
            if term or trunc:
                break
        return {**score.result(), "tier": lvl["tier"], "seed": lvl["seed"]}
    finally:
        env.close()


class AgentPolicy:
    """Adapter for the real submission agent (uses observations only)."""

    def __init__(self):
        import sys
        from doom_env.spaces import action_space, observation_space
        other = os.environ.get("DOOM_AGENT_DIR")  # score another checkout's agent on the same levels
        if other:
            import importlib.util
            sys.path[:0] = [str(Path(other) / "submission"), other]
            spec = importlib.util.spec_from_file_location("other_agent", Path(other) / "submission" / "agent.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.agent = mod.DoomAgent(observation_space(), action_space(), "cpu")
            return
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
        from submission.agent import DoomAgent
        self.agent = DoomAgent(observation_space(), action_space(), "cpu")

    def reset(self, env):
        self.agent.reset()

    def act(self, obs, env):
        return self.agent.act(obs)


def make_policy(name: str):
    if name == "teacher":
        from training.teacher import Teacher
        return Teacher()
    return AgentPolicy()


def _low_priority() -> None:
    """Worker processes run below normal priority so a laptop stays responsive."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        else:
            os.nice(10)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")


_POLICY = {}


def _work(job):
    name, seed = job
    if name not in _POLICY:
        _POLICY[name] = make_policy(name)
    return run_episode(make_level(seed), _POLICY[name])


def summarize(rows: list[dict]) -> str:
    out = []
    total = 0.0
    for tier in sorted({r["tier"] for r in rows}):
        rs = [r for r in rows if r["tier"] == tier]
        mean = lambda k: float(np.mean([r[k] for r in rs]))
        total += TIER_WEIGHT.get(tier, 0.0) * mean("score")
        out.append(f"tier {tier}: n={len(rs):3d} score {mean('score'):6.1f}  exit {mean('exit'):.2f}  dead {mean('dead'):.2f}  "
                   f"keys {mean('keys'):.2f}  doors {mean('doors'):.2f}  locked {mean('locked'):.2f}  explore {mean('explore'):.2f}  kills {mean('kills'):.1f}")
    out.append(f"tier-weighted score: {total:.1f}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("teacher", "agent"), default="teacher")
    ap.add_argument("--seeds", default=f"{HELD_OUT}:{HELD_OUT + 16}")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.seeds.split(":"))
    policy = make_policy(args.policy) if args.workers <= 1 else None
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=_low_priority) as pool:
            rows = pool.map(_work, [(args.policy, s) for s in range(lo, hi)], chunksize=1)
    else:
        rows = [run_episode(make_level(seed), policy) for seed in range(lo, hi)]
    if args.verbose:
        for r in rows:
            print(r)
    print(summarize(rows))
    print(f"{sum(r['tic'] for r in rows) / 4 / (time.time() - t0):.0f} decisions/s")


if __name__ == "__main__":
    main()
