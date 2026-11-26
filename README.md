# Can It Play DOOM? — Yugaantar DevForge, Track 1

A neural agent that plays unseen DOOM levels: it explores, fights, finds keycards, opens locked doors and
looks for the exit. Built from the official `devforge-doom-template` during the hackathon.

Every action comes from the trained network in `submission/`. There is no scripted decision logic,
pathfinding or aim maths in the submitted agent.

## Approach in one paragraph

Pure reinforcement learning rarely discovers keycard → locked door → exit in a few hours. We first train
the network to imitate a **training-only scripted teacher** with DAgger (the network plays, the teacher
labels every state it visits, so the network learns to recover from its own mistakes). The agent keeps its own **SLAM-style map** of the level, built only
from its observations and its own actions, and reads it through a **learned planner module**.

## Door colours from the frame

The rays report "locked door" but not which key it needs, so an agent with one key wastes time at
doors it cannot open. `submission/slam.py` reads each locked door's colour (red, blue or yellow
grille) from the frame, using the depth buffer to pick the pixels that belong to that door. Until
the agent holds the matching key, that door is shown to the network as a wall, in both its rays and
its map; it turns back into a locked door when the key is picked up. This is observation
preprocessing only. On 192 held-out levels with the same weights it raised the local score from
1073 to 1108 (tier 3 exits 42% -> 50%, tier 4 12% -> 15%; tiers without locks unchanged).

## Repository layout

```
submission/agent.py      DoomAgent: the evaluation entry point (network + safety guard)
submission/model.py      policy network and observation encoding
submission/slam.py       occupancy map + odometry from the 64 rays and the agent's own actions
submission/actions.py    the 23 button combinations the policy chooses from
submission/weights/      trained weights (Git LFS)

training/mapgen.py       procedural maps in the organisers' ASCII-grid UDMF format, all four tiers
training/teacher.py      scripted teacher (TRAINING ONLY): frontier exploration, keys, doors, combat
training/dagger.py       imitation learning (DAgger)
training/scorer.py       local estimate of the official scoring, from privileged state
training/evaluate.py     score a policy on a held-out map set
training/compare.py      score any checkout's agent on the same maps and seeds
training/watch.py        watch the agent play, with its SLAM map and objectives panel

env/ docs/ tests/ validate.py   the official template, unchanged
```

## The agent

| Module | Design |
|---|---|
| Rays | circular 1D-CNN over the 64 rays (distance, near-ness, hit type) |
| Entities | per-entity MLP + type embedding, masked attention pooling and max pooling |
| Vitals | MLP over normalised vitals and the agent's previous action |
| Spatial memory | `slam.py`: heading from the agent's own turn actions (Doom's turn ramp, measured), position from the observed velocity, rays painted into an occupancy grid; the network reads a fine 32×32 ego-centric crop (free, wall, door, locked door, visited, frontier) and a zoomed-out 16×16 view |
| Learned planner | a small conv block applied 8 times over the map (value-iteration style) so information spreads across it; all weights trained |
| Memory | GRU (256) |
| Heads | "which way to go" (16 directions + none), 23 button combinations conditioned on it, value, and helper heads (teacher's goal type, locked door seen) |

About 1.5M parameters; about 1–4 ms per decision on one CPU thread (limit: 40 ms). Inference always
runs on CPU.

What is not learned in the submitted agent, and why it is allowed:
- **Action masking** of `USE` and weapon-switch on the step after they were pressed (both are
  edge-triggered; masking from the agent's own previous action).
- **Observation preprocessing**: encoding, and the map built from observations and the agent's own actions.
- **A safety guard**: malformed input or any internal error returns a valid action instead of crashing.

## Reproduce

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group train                              # official env + PyTorch
uv run python validate.py                          # official validator

uv run python training/make_maps.py                # training maps + held-out maps
uv run python training/evaluate.py --policy teacher                # score the teacher
uv run python training/dagger.py --rounds 8 --device mps           # imitation (use --device cpu or cuda elsewhere)
uv run python training/publish.py runs/dagger/best.pt              # copy weights into submission/
uv run python training/watch.py                                    # watch the submitted agent
uv run python training/watch.py --live                             # follow the newest training checkpoint
```

## Rules compliance

- Decisions come from trained neural parameters only. The teacher (which uses path search and aim
  maths with privileged engine state) lives in `training/` and is never imported by `submission/`.
- No Stable-Baselines3 or other off-the-shelf code: the network and the DAgger pipeline are our own.
- The agent uses only the official observation dictionary; no engine, network or disk access.
- Training maps are generated; nothing is memorised from evaluation.
