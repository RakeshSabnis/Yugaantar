# Can It Play DOOM? — Project brief (draft)

<!-- Numbers marked TODO are filled in from the final local evaluation. Target: 2 pages as PDF. -->

## USP: teach first, then let it learn

Pure reinforcement learning rarely discovers keycard → locked door → exit in hackathon time. We first
train the network to imitate a **training-only scripted teacher** that has full engine access, using
**DAgger** so it also learns to recover from its own mistakes, then **fine-tune with our own recurrent PPO**
whose rewards mirror the scoring categories. The submitted agent is purely neural: it sees only the official
observations and every action comes from the network.

Two things make the imitation learnable and honest:
- **The teacher only acts on what it has seen.** It keeps a line-of-sight "seen" mask and explores frontiers,
  instead of walking straight to a hidden key. The student can therefore copy it from partial observations.
- **Maps in the organisers' own format.** We decoded the organisers' 13 test maps (UDMF drawn from ASCII
  grids, 64-unit cells) and wrote a generator that produces unlimited random maps in that style across all
  four tiers, each checked for solvability (key reachable before its lock, exit only behind the lock).

## Architecture (1.1M parameters, ~0.5 ms per decision on one CPU thread)

| Module | Design |
|---|---|
| Ray encoder | Circular 1D-CNN over the 64 rays (distance, near-ness, hit-type one-hot); wraps around 360° |
| Entity encoder | Per-entity MLP with a type embedding, then masked attention pooling (2 learned queries) + max pooling: any number of enemies/items |
| Vitals | MLP over normalised vitals plus the agent's own previous action |
| Memory | GRU (256) over the fused embedding, reset before every map |
| Heads | 23 discrete button combinations, value head, and **helper heads**: the teacher's current goal (exit / key / frontier) and "locked door seen this episode", which force the memory to keep task-relevant facts |

Safety: inputs sanitised (NaN/inf/garbage), `act()` guarded so it always returns a valid action, inference
pinned to CPU with a warm-up pass in `__init__`. Edge-triggered buttons (USE, weapon switch) are masked
after being pressed, as the contract requires a release between presses. Low-temperature sampling avoids
deterministic loops.

## Training pipeline

1. **Map generator** (`training/mapgen.py`): arenas (tier 1), multi-room mazes with dead ends and doors
   (tier 2), keycard → locked door → exit chains (tier 3), two-lock gauntlets (tier 4). 600 training maps,
   60 held-out maps never trained on, plus Freedoom levels as a different-style check.
2. **Local scorer** (`training/scorer.py`): our estimate of the official categories (exit with health/time
   bonus, keys, doors, locks, explored areas, kills, damage, early death), computed from privileged state.
3. **Teacher** (`training/teacher.py`, training only): frontier exploration, key → lock → exit planning,
   door handling, combat. Held-out score: TODO.
4. **DAgger** (`training/dagger.py`): round 0 teacher-driven, then the network drives with a decaying
   teacher mix while the teacher labels every visited state; data aggregated across rounds.
5. **Recurrent PPO** (`training/ppo.py`): our own implementation; rewards mirror the scorer; a fading
   cross-entropy term to the teacher's labels prevents forgetting; checkpoints kept only if they beat the best
   on held-out maps.

## Results (local held-out maps, our scorer)

| Policy | Weighted | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---|---|---|---|---|
| Random | −54 | | | | |
| Teacher (privileged, training only) | 1242 | | | | |
| Imitation (DAgger) | TODO | | | | |
| + PPO | TODO | | | | |

## Limitations and next steps

TODO: egocentric map memory, curiosity (RND), map-style mismatch risk, what we would do with more time.
