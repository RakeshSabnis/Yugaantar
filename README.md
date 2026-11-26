# Can It Play DOOM? — Yugaantar 2026 · DevForge Track 1

A recurrent neural agent that plays unseen DOOM levels from the official `doom_env` observations:
it explores, fights, opens doors, finds keycards, unlocks locked doors and heads for the exit.
Every action comes from the trained network in [`submission/`](submission/).

Built from the official starter template (`env/`, `validate.py`, `tests/`, `docs/`, `Dockerfile` are
unchanged template files).

## How it works

```
observation ──┬─ 64 rays ───────────────► circular 1D CNN ─────┐
              ├─ visible entities ──────► MLP + attention pool ─┤
              ├─ vitals + last action ──► MLP ──────────────────┼─► GRU ─► 9 button logits
              └─ rays + own motion ─► spatial memory ─► 2D CNN ─┘        (+ value head)
                                      (mapper.py)
```

| Part | File | What it does |
| --- | --- | --- |
| Spatial memory | [`submission/mapper.py`](submission/mapper.py) | The observation has no position or heading, so the agent dead-reckons its own pose (heading from its turn buttons with a tic-accurate model of the engine; position from the velocity in `vitals`) and paints what the rays reveal into an occupancy grid: free space, walls, doors, locked doors, where it has walked, keys seen, exit seen. The network reads two egocentric crops of that grid (±512 and ±2048 units, forward = up), which is how it remembers a locked door it passed earlier. |
| Policy network | [`submission/model.py`](submission/model.py) | Ray CNN (wraps around 360°), attention pooling over any number of entities, map CNN, GRU core, one output per button. About 1M parameters. |
| Agent | [`submission/agent.py`](submission/agent.py) | Loads the weights, keeps the memory, samples buttons from the network. Around it: `USE` debouncing and opposite-button masking (allowed action masking), CPU-only inference, and a guard so a fault can never crash a map. |

The spatial memory is preprocessing only: it never chooses an action.

## How it was trained

1. **Procedural levels** ([`training/mapgen.py`](training/mapgen.py)): random levels for all four
   tiers in the same grid → UDMF format as the organizers' test maps, each checked by a solver.
   Room shapes, corridor loops, cell scale, monster mix, key colours and start angle are randomized.
   One training level in five is a hand-made Freedoom level instead, so the agent does not depend
   on one map style.
2. **Teacher, training only** ([`training/teacher.py`](training/teacher.py)): a scripted explorer
   that plans on the same ray-built memory the network gets, so everything it does is learnable
   from the legal observations. It is never imported by `submission/`.
3. **Imitation with DAgger** ([`training/dagger.py`](training/dagger.py)): the teacher labels every
   step while the network drives more and more of each episode, which teaches it to recover from
   its own mistakes. After every round the network is scored alone on held-out levels.

Details: [`training/README.md`](training/README.md).

## Results

Network playing alone on 48 held-out generated levels (seeds never used in training), scored with
our local stand-in for the hidden scorer ([`training/scorer.py`](training/scorer.py); the weights
are our guess, so the numbers are relative):

| | Tier 1 exit | Tier 2 exit | Tier 3 exit | Tier 4 exit | Tier-weighted score |
| --- | --- | --- | --- | --- | --- |
| Scripted teacher (ceiling for imitation) | 50% | 58% | 58% | 17% | 1079 |
| Network after DAgger round 1 | 25% | 17% | 17% | 0% | 473 |
| Network after DAgger round 2 (committed weights) | 17% | 25% | 25% | 0% | 584 |

Inference: 7.6 ms median per decision on one CPU thread (budget 40 ms), 2.1 s initialization.

## Known limitations

- The exit pad is found from the frame (its red floor texture, placed with the depth buffer). On a
  map with a different exit style the agent falls back to exploring until it walks over the exit;
  exits that are wall switches are not handled.
- Lifts and switch-operated doors are not handled.
- Tier 4 (several locks, more monsters) is the weakest tier for both teacher and network.

## Repository layout

```
submission/   agent.py, model.py, mapper.py, weights/policy.pt (Git LFS)   <- what is evaluated
training/     mapgen, teacher, dagger, scorer, play, watch, aws_run.sh     <- how it was trained
env/          official doom_env wrapper (template)
maps/ tests/  template test map and wrapper tests
docs/         template docs (CONTRACT, RULES, SUBMISSION, EVALUATION) and the problem statement
validate.py   official local validator (template)
```

## Run it

```bash
uv sync --group train
uv run python validate.py                                        # official validator
uv run python -m training.play --policy agent --seeds 1000000:1000048 --workers 4   # score the agent
uv run python -m training.watch --policy agent --seed 1000002    # watch it play, with its memory map
uv run python -m training.dagger --smoke                         # check the training pipeline
```

## Rules compliance

- Actions come from trained parameters only. No pathfinding, rule trees or aim maths in
  `submission/`; the scripted teacher lives in `training/` and is used only to generate labels.
- No Stable-Baselines3 or other off-the-shelf policy: the network, memory and training loop are
  our own.
- The agent uses only the observation dict, makes no network or file-system calls, and runs on CPU.
