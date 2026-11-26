# Can It Play DOOM? — Team TriTokens · Yugaantar DevForge, Track 1

A neural agent that plays unseen DOOM levels: it explores, fights, finds keycards, opens locked doors and
looks for the exit. Built from the official `devforge-doom-template` during the hackathon.

Every action comes from the trained network in `submission/`. There is no scripted decision logic,
pathfinding or aim maths in the submitted agent.

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). Weights are stored with Git LFS
(`git lfs install` once, before cloning).

```bash
uv sync --group train                                        # install the official environment and PyTorch
uv run python validate.py                                    # the organisers' validator: must print PASSED
uv run python training/watch.py --map maps/two_rooms.wad     # watch the agent play, with its map
```

`watch.py` opens two windows: the game from the agent's eyes, and the map it is building as it
explores. To watch it on generated levels of every tier:

```bash
uv run python training/make_maps.py                          # generate training and held-out levels (once)
uv run python training/watch.py                              # a random held-out level each time
uv run python training/watch.py --policy teacher             # the training-only teacher, for comparison
```

Useful options: `--speed 2` plays twice as fast, `--no-map` hides the map window.

| I want to... | Go to |
| --- | --- |
| see what is evaluated | `submission/agent.py` |
| understand the network | `submission/model.py`, and "The agent" below |
| understand the map memory | `submission/slam.py` |
| see how it was trained | `training/README.md` |
| read the 2-page brief | `docs/brief.pdf` |

## Approach in one paragraph

Pure reinforcement learning rarely discovers keycard → locked door → exit in a few hours. We first train
the network to imitate a **training-only scripted teacher** with DAgger (the network plays, the teacher
labels every state it visits, so the network learns to recover from its own mistakes). The same loop carries a
**reinforcement-learning term with a curiosity reward** (below). The agent keeps its own **SLAM-style map** of the level, built only
from its observations and its own actions, and reads it through a **learned planner module**.

## Reinforcement learning and curiosity in training

`training/dagger.py` is not pure imitation. Every game the network plays is also scored with a shaped
reward that mirrors the scoring categories (exit, keys, doors, locks, kills, minus damage, death and
time), plus an **intrinsic curiosity reward for entering a cell of its own map that it has never stood
in**. The value head regresses the return, and actions that did better than the value predicted are
reinforced (self-imitation, `--sil`); an actor-critic policy-gradient term (`--pg`) and a learned
novelty bonus (random network distillation, `training/rnd.py`) are in the same loop. Everything here
runs in training only; the submitted agent contains none of it.

Stand-alone PPO fine-tuning of the imitation policy lowered the held-out score in every configuration
we tried, so it is not part of this repository.

## Results of the submitted agent

Levels in the organisers' sample style (64-unit cells, same doors, keys and exit pad), network playing
alone, 8 to 12 seeded runs each:

| Level | Exit reached |
| --- | --- |
| Organisers' sample layout: key, locked door, exit | 8 / 8 |
| Four rooms, three doors (tier 2 style) | 8 / 8 |
| Key in a side room, locked door, exit (tier 3 style) | 8 / 8 |
| Two keys, two locks, monsters (tier 4 style) | 11 / 12 |
| Deeper gauntlet, two locks, more monsters (tier 4 style) | 8 / 12 |

Stress test on 384 much larger generated levels never used in training (routes up to 100 cells): the
agent picks up the key in 95% of tier 3 levels and opens the locked door in 81%, and averages 1.7 keys
and 1.4 locked doors per tier 4 level. Our local score rose from 1083 to **1198** with the three
additions described below, using the same network weights.

## Door colours from the frame

The rays report "locked door" but not which key it needs, so an agent holding one key can waste time
at doors it cannot open. `submission/slam.py` reads each locked door's colour (red, blue or yellow
grille) from the frame, using the depth buffer to pick the pixels that belong to that door. Until
the agent holds the matching key, that door appears in its rays and its map as a wall; it turns back
into a locked door when the key is picked up.

## Exit pads from the frame, and jam recovery

- **Exit pads.** Neither rays nor entities show the exit. `submission/slam.py` finds the exit pad's red
  floor in the frame, places it with the depth buffer, and writes those cells into the map as a place
  worth walking to. Upright red things (blood, corpses, fireballs) are rejected because their pixels
  share one depth, and frames tinted by damage are skipped.
- **Jam recovery.** Doors are 64 units wide and the player 32. If the agent has pressed forward for
  several steps without moving and is not waiting at a door, forward is withheld for a few steps and
  the network's next choice (side-step, turn, back) frees it.

## Repository layout

```
submission/                 what is evaluated
  agent.py                  DoomAgent: the evaluation entry point
  model.py                  policy network and observation encoding
  slam.py                   map and odometry built from the rays, the frame and the agent's own actions
  actions.py                the 23 button combinations the policy chooses from
  weights/policy.pt         trained weights (Git LFS)

training/                   how it was trained (see training/README.md)
  mapgen.py, make_maps.py   procedural levels in the organisers' format, all four tiers
  teacher.py                scripted teacher, training only
  dagger.py                 imitation (DAgger) with the reinforcement-learning term and curiosity reward
  rnd.py                    learned novelty bonus (random network distillation)
  scorer.py, evaluate.py    local estimate of the official scoring; score a policy on held-out levels
  diagnose.py               per-tier failure analysis: how far each run got and where its time went
  publish.py                copy a checkpoint into submission/weights/
  watch.py                  watch the agent play, with its map

docs/brief.pdf              the 2-page project brief
docs/*.md, env/, tests/, maps/, validate.py, Dockerfile, docker/    the official template, unchanged
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

## Train it yourself

```bash
uv run python training/make_maps.py                            # training and held-out levels
uv run python training/evaluate.py --policy teacher            # score the teacher on held-out levels
uv run python training/dagger.py --rounds 8 --device cpu       # train (cpu, cuda or mps)
uv run python training/publish.py runs/dagger/best.pt          # copy the best checkpoint into submission/
uv run python training/evaluate.py --policy agent              # score the trained agent
uv run python training/diagnose.py --tier 4 --policy agent     # where do runs on a tier break down?
uv run python training/watch.py --live                         # follow the newest checkpoint while training
```

Each stage is explained in `training/README.md`.

## Rules compliance

- Decisions come from trained neural parameters only. The teacher (which uses path search and aim
  maths with privileged engine state) lives in `training/` and is never imported by `submission/`.
- No Stable-Baselines3 or other off-the-shelf code: the network and the DAgger pipeline are our own.
- The agent uses only the official observation dictionary; no engine, network or disk access.
- Training maps are generated; nothing is memorised from evaluation.
