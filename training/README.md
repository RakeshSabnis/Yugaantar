# Training pipeline

Everything in this folder runs at training time only. The submitted agent is `submission/agent.py`,
and it imports nothing from here.

## Stages

1. **Levels** (`mapgen.py`, `make_maps.py`). ASCII grids are turned into UDMF WADs in the same format
   as the organisers' test maps: arenas (tier 1), multi-room mazes with doors and dead ends (tier 2),
   key -> locked door -> exit chains (tier 3) and multi-lock gauntlets (tier 4). Sizes, monsters, key
   colours and door placement are randomised, every level is checked for solvability, and a held-out
   set is never trained on.
2. **Teacher** (`teacher.py`). A scripted player used only to label states: frontier exploration,
   key and lock planning, door handling, combat.
3. **Imitation with DAgger** (`dagger.py`). The network plays; the teacher labels every state the
   network visits; data is aggregated across rounds, with later rounds sampling the key and gauntlet
   tiers more heavily. This is what teaches the network to recover from its own mistakes.
4. **Reinforcement-learning term** (`dagger.py`, `--sil`, `--pg`). Each game the network plays is
   also scored with a shaped reward that mirrors the scoring categories (exit, keys, doors, locks,
   kills, minus damage, death and time). The value head regresses the return, and actions that did
   better than the value predicted are reinforced.
5. **Curiosity** (`dagger.py`, `rnd.py`). An intrinsic reward for entering a cell of the agent's own
   map for the first time, plus a learned novelty bonus (random network distillation).

## Measuring

- `scorer.py`: our estimate of the official categories, computed from privileged state.
- `evaluate.py`: score the teacher or the agent on a held-out level set.
- `diagnose.py`: for one tier, how far each run got (which key, which lock), how it ended, and where
  the time went. This is what pointed us at locked-door colours.
- `watch.py`: watch the agent play, with its map.

## Run

```bash
uv sync --group train
uv run python training/make_maps.py                                  # training and held-out levels
uv run python training/evaluate.py --policy teacher                  # score the teacher
uv run python training/dagger.py --rounds 8 --device cpu             # train (cpu, cuda or mps)
uv run python training/publish.py runs/dagger/best.pt                # copy weights into submission/
uv run python training/diagnose.py --tier 4 --policy agent           # failure analysis
uv run python training/watch.py                                      # watch the submitted agent
```
