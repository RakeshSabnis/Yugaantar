# Training pipeline

Everything here is training-time scaffolding. The evaluated agent is `submission/agent.py`, which
acts only from the trained network in `submission/model.py`.

| File | What it does |
| --- | --- |
| `mapgen.py` | ASCII grid -> UDMF WAD (same format as the organizers' fixtures) and random levels for tiers 1-4. A solver rejects unsolvable layouts. Cell scale, room shapes, loops, monster mix, key colours and start angle are randomized. Seeds >= 1,000,000 are held out and never trained on. |
| `scorer.py` | Local stand-in for the hidden scorer (exit > keys/doors > exploration > kills, minus damage/death). The weights are our guess. |
| `teacher.py` | Scripted explorer used only to produce demonstrations. It plans on the same ray-built memory the network gets, so what it does is learnable; its one privilege is the true pose. |
| `dagger.py` | Imitation stage: the teacher labels every step while the network drives more and more of each episode; retrain; score the network alone on held-out levels. |
| `play.py` | Score the teacher or the submission agent on generated levels. |
| `watch.py` | Live viewer: first-person view next to the agent's memory map. |
| `aws_run.sh` | One-command run on a fresh Ubuntu GPU instance. |

## Spatial memory (`submission/mapper.py`)

The observation has no position or heading. The agent dead-reckons its pose (heading from its own
turn buttons with a tic-accurate model of the engine's turning: 0 degrees of error over a full
episode; position from the velocity in `vitals`: about 50 units after 1,050 steps) and paints ray
hits into a 16-unit occupancy grid: free, wall, door, locked door, walked, key seen, exit seen. The
network receives two egocentric crops of it (+-512 and +-2048 units, forward = up). The exit pad is
not in the rays or the entity list, so the "exit seen" layer comes from the frame: saturated-red
floor pixels placed with the depth buffer.

## Run

```bash
uv sync --group train
uv run python -m training.play --policy teacher --seeds 1000000:1000024 --workers 4   # teacher quality
uv run python -m training.watch --seed 1000001                                         # watch it
uv run python -m training.dagger --smoke                                               # pipeline check
bash training/aws_run.sh                                                               # full run (AWS)
```

## Status and next steps

- Done: generator, scorer, teacher, memory, network, agent, imitation pipeline, viewer.
- Not done yet: the full imitation run (AWS), PPO fine-tuning with a curiosity bonus on top of the
  imitation weights, mixing Freedoom/community WADs into training as a different-style check.
