# Track 1 work log

What we tried, what worked, what didn't, and what we went ahead with. Newest entries are at the bottom.
Scores are from our own local scorer (`training/scorer.py`), an estimate of the official categories. They
compare our own checkpoints with each other; they are not leaderboard scores.

**How to read the scores**
- "60 maps" = the first held-out set. "180 maps" = both held-out sets together (used from 21:45 on, because
  60 maps turned out to be too noisy). Numbers from different sets are not comparable.
- The weighted score is: tier-1 average + 1.5 × tier-2 + 2 × tier-3 + 3 × tier-4 (per-map averages).
- A single map tops out at roughly 140–270 points, so a perfect weighted score would be about 1,645.

## Current state (final, 02:10)

| | |
|---|---|
| Team submission | **`main`**: the network trained here, plus image-based door colours, exit detection and a jam mask added on `main` |
| `main` vs `yash`, 180 maps | 884 vs 762 |
| `main` vs `yash`, 80 maps from `main`'s generator | 1,081 vs 971 |
| This branch (`yash`) | The training record: imitation + self-imitation RL + map curiosity; 860 / 762 / 805 on three seed sets of the 180 maps |
| Brief | `team/Project_Brief_Track1.pdf` (describes the `yash` agent) |

---

## Timeline

### 16:00–16:35 · Setup, maps, scorer, teacher

| Tried | Result | Kept? |
|---|---|---|
| Read the track statement, template docs, wrapper and validator | Found the organisers' test maps are UDMF drawn from ASCII grids (64-unit cells) | Yes: basis for our map generator |
| Moved the template to the repo root | Scorer expects `submission/agent.py` at the root; also found `env/` was never committed (ignored by a `.gitignore` rule on macOS) | Yes |
| Own grid → UDMF map generator, four tiers | Our door/key/exit maps behave exactly like the organisers' fixtures | Yes |
| Local scorer from `privileged_state()` | Works; 60 held-out maps scored in 7 s | Yes |
| Scripted teacher (frontier exploration, keys, doors, combat) | First version: 850 | Yes |
| Teacher bug: pressed USE from a diagonal cell and waited forever | Fixed → 1,054 | Yes |
| **Generator bug: locked door placed next to a plain door can never open** (a Doom door only rises to its lowest neighbouring ceiling) | Fixed → teacher 1,242 | Yes |
| Teacher detours for health and a shotgun | 1,226, slightly worse | **No, reverted** |

### 16:35–17:10 · First neural agent (imitation)

| Tried | Result | Kept? |
|---|---|---|
| Network: ray CNN + entity attention + vitals MLP + GRU, 23 button combinations, helper heads | 1.1M parameters, 0.4 ms per decision | Yes |
| Training on the Mac's GPU instead of CPU | 14× faster | Yes |
| DAgger imitation, 10 rounds planned | 95 → 444 → 549 → **668** (round 3), then 651, 318 | Kept round 3; stopped the run when it degraded |
| First weights committed through Git LFS | Validator passes | Yes |

### 17:10–18:35 · Heat, first RL attempt, SLAM map

| Tried | Result | Kept? |
|---|---|---|
| Moved training to efficiency cores, fewer parallel games (Mac ran warm) | Cooler, about 2× slower; later relaxed to 4–5 games at normal priority | Yes |
| PPO fine-tune (first attempt) | 668 → 168 → 13 within two iterations | **No** |
| PPO with one optimizer step per minibatch and a value warm-up | Still fell: 668 → 494 → 313 | **No** |
| **SLAM map** from the 64 rays: heading from our own turn actions, position from observed velocity | Turn model exact on 99.8% of steps; position drift about a quarter of a cell (median) | Yes |
| Scan matching to correct heading | Made heading much worse (median error 31°) | **No, removed** |
| Fitted velocity integration (2.589 × v_now + 1.801 × v_prev) | Median position error 61 → 18 units | Yes |
| Teacher sidesteps incoming projectiles | Combat deaths 40% → 33% | Yes |
| Network with the map as input | 285 → 366 → **674** (60 maps) | Yes |
| Sampling temperature sweep (0.15 / 0.3 / 0.5 / 0.8) | 361 / 472 / 674 / 547 | Kept 0.5 |
| Lower temperature while an enemy is visible | 732 on 60 maps, but only +1% on 120 maps | **No: removed at 20:10** as a borderline hand-written condition |

### 18:35–20:35 · Frontier, wide view, planner

| Tried | Result | Kept? |
|---|---|---|
| Frontier layer in the map (seen floor next to unexplored space) | 685 on 60 maps; no clear gain alone | Kept as an input |
| Zoomed-out 16×16 map view | 633, 575; no gain alone | Kept as an input |
| Weight averaging of two rounds | 659 vs 759 for the better round | **No** |
| Ported `main`'s aiming idea into our teacher: turn only above 7.5°, side-step for fine aim, hold fire until lined up, target priority | Teacher 1,241 → 1,305 | Yes |
| Firing during the fine side-step (variant) | No measurable difference on 180 maps | **No** |
| **Learned planner** (conv block applied repeatedly over the map) + "which way to go" head feeding the button head | First version made training 40× slower | Shrunk it (half resolution, 8 passes) |
| Planner network, imitation rounds | 436 → 457 → 585 → 706 → **834** (60 maps); 734 on 180 maps | **Yes: became the submission** |
| Compared with `main` on the same 180 maps, same seeds | Ours 649 (at the time), `main` 489 | — |

### 20:35–21:45 · Reinforcement learning (PPO)

| Tried | Result | Kept? |
|---|---|---|
| PPO from the planner agent (lr 5e-5) | 834 → 384 (60 maps) | **No** |
| PPO, 5× smaller updates, 60 games per update | Crashed: GPU out of memory | Fixed (values computed in slices) |
| Same, after the fix | 834 → 480 | **No** |
| **Bug found:** the teacher term in PPO was computed at the rollout temperature, pushing the policy flatter than imitation had learned | Fixed | Yes (fix kept) |
| PPO after that fix | 834 → 676 on 60 maps; 684 → 563 on 120 maps | **No**: smaller drop, still a drop |

Conclusion: at this scale PPO's signal is too noisy (the value estimate has had minutes, not hours, of
training) and it overwrites what imitation learned. PPO stays in the repo (`training/ppo.py`) but is not
in the submitted weights.

### 21:45–22:40 · More imitation, measured on 180 maps

| Tried | Result | Kept? |
|---|---|---|
| Scoring every round on all 180 held-out maps | Less noisy, comparable between runs | Yes |
| Planner network, further rounds | 579 → 712 → **759** → 624 → 694 → **825** | **Yes: 825 is the submission** |
| "Stuck" metric (share of steps without movement) | Agent 9–15% at 825 (was 10–26%); teacher 8–15% | Yes |

Lesson: restarting a run throws away its collected games, so the first round after a restart always dips.

### 22:40–23:20 · Gauntlet analysis and what it changed

Ran teacher and agent on all 45 held-out gauntlet maps (`training/diagnose.py`).

| Stage reached | Teacher | Agent (825) |
|---|---|---|
| Key 1 | 34, at step ~300 | 29, at step ~490 |
| Locked door 1 opened | 34 | 14 |
| Key 2 | 32 | 10 |
| Locked door 2 opened | 30 | 4 |
| Exit | 26 | 2 |
| Died / ran out of time | 16 / 3 | 12 / 31 |

Findings:
1. In 15 of 45 runs the agent picks up a key and never opens a lock. In its map a locked door looked the
   same with or without a key.
2. It spends 11% of failed runs at doors it cannot open, and does not remember having tried them.
3. It is slow: about 490 steps to the first key against the teacher's 300 (limit 1,050).

| Tried in response | Result | Kept? |
|---|---|---|
| Focused run on a 1,500-map set weighted to keycard/gauntlet maps | 709, 686 after two rounds | Stopped to add the fixes below; the map set and weighting are kept |
| "Holding a key" layer in both map views | Verified on test maps | In training now |
| Memory of locked doors tried and still shut (cleared on a new key) | Verified on test maps | In training now |
| **Map bug: the SLAM grid was offset half a cell from Doom's grid** (door marks landed in the agent's own cell and were wiped) | Fixed | In training now |
| Open doors are removed from the map | Verified | In training now |
| **Self-imitation RL inside the imitation loop**: value head regresses the return; actions that did better than expected are reinforced | Smoke test passes | In training now |
| **Curiosity reward from the agent's own map**: bonus for entering a map cell it has never stood in | Part of the RL reward | In training now |

### 23:20–00:05 · Comparison with `main`, first RL agent

Both branches scored from clean checkouts of what is pushed, same maps, same seeds, same scorer
(`training/compare.py`).

| Test set | `main` | `yash` (825 agent) | Lead |
|---|---|---|---|
| 180 maps from our generator | 475 | 790 | +66% |
| 80 maps from `main`'s generator (other cell sizes, random start directions) | 561 | 973 | +73% |

`yash` is ahead on score and exit rate in every tier on both sets. `main` is better at two things: it dies
less (0–15% vs 10–25% on its own maps) and gets stuck less (4–6% of steps vs 8–16%).

| Tried | Result | Kept? |
|---|---|---|
| Run with imitation + self-imitation RL + curiosity + key-aware map | 745 → 735 → 740 → **860** (180 maps) | Yes |
| Checked the 860 checkpoint on a second seed set and on `main`-style maps | 762 (previous agent 790) and 971 (previous 973): a tie overall | **Yes: submitted**, because at equal score it is the agent that actually contains the RL stage, the curiosity reward and the fixed map |
| Gauntlet analysis on the new agent | Lock 1 opened 14 → 17, key 2 10 → 14, lock 2 4 → 8, exits 2 → 3; but time at doors it cannot open 11% → 22%, deaths 12 → 17 | The tried-door memory is not yet used well; training continues |

Lesson: one score on one seed set is not enough to pick a checkpoint. The same agent moves by about ±40
between seed sets.

### 00:05–01:35 · "A more proper RL", late fixes, brief

| Tried | Result | Kept? |
|---|---|---|
| Further rounds of the imitation + self-imitation run | 805, 759 after the 860 round: a plateau around 780 | Stopped |
| **PPO with a trained critic** (value head from the previous rounds), map curiosity and a learned curiosity module (random network distillation) | Critic error 0.07 from the start (earlier PPO runs: about 1.0), yet 860 → 628 after six policy updates | **No.** Fourth PPO failure |
| Why PPO fails here | Not the RL signal or the critic. Every PPO run trains on a small fresh batch and discards it; the imitation runs show the same dip whenever they restart with an empty pool. This agent is only stable on a large accumulated pool of games | Lesson kept |
| **Actor-critic term inside the pooled loop** (signed advantage on each round's own games) + **learned curiosity (RND)** added to the reward | Runs stably | In the last run |
| Penalty for standing at a locked door the agent cannot open, penalty for not moving, reward for newly *seen* map cells, 3× teacher weight where the agent is stuck or blocked | Stuck time 8–14% in the first round (submitted agent 9–22%) | In the last run |
| Teacher taps fire at range (organisers' note on pistol spread) | 2,329 vs 2,426 over 180 maps | **No**, left off |
| Organisers' checklist (weights via LFS, limits, self-contained `submission/`, public repo, validator) | All pass | — |
| 2-page brief | Written, exported with the browser's print-to-PDF, checked at 2 pages | Yes |

Per-tier picture of the submitted agent across the three test sets (exit rate): combat 33–60%,
exploration 42–65%, keycard 51–71%, gauntlet 9–15%.

### 01:35–02:10 · Last attempts and the final comparison with `main`

| Tried | Result | Kept? |
|---|---|---|
| Last run (actor-critic term, learned curiosity, blocked/stuck penalties), 4 rounds | 760 → 728 → 799 → 803 on 180 maps; submitted agent 860 on the same seeds | **No**, stopped at the freeze |
| Ensemble of 2 networks (probabilities averaged) | 180 maps: 832 vs 860 (seed set A), 851 vs 805 (fresh seeds); neutral maps 869 vs 943 | **No**: mixed. Support for it stays in `agent.py`; with one weights file it behaves exactly as before |
| Ensemble of 3 networks | 786 vs 860 | **No** |

**Final comparison with `main` (its 00:53 version), frames on, same maps and seeds, clean checkouts:**

| Test set | `main` | `yash` | |
|---|---|---|---|
| 180 maps from our generator | 884 | 762 | `main` +16% |
| 80 maps from `main`'s generator | 1,081 | 971 | `main` +11% |

`main` is better on both sets and in every heavily weighted tier; on gauntlets it reaches the exit twice as
often (30% vs 15% on its own maps) and is stuck half as much (8–9% vs 17%).

What `main` is: **the same trained network as `yash`** (the weights files are byte-identical), plus three
additions around it: locked-door colours read from the camera image (doors the agent cannot open yet are
shown to the network as walls), exit pads read from the image and marked in the map, and a mask that
withholds "forward" for a few steps when the agent pushes without moving.

What we got wrong on `yash`: we treated a lock's colour as unobservable and tried to train around it with a
"tried door" memory that was only half learned. The colour is in the camera image. We had dropped pixels
at the start and did not come back to them for the one fact the rays do not carry.

**Conclusion: `main` is the team's submission.** `yash` is the record of how the network was trained.

---

## What we are going ahead with

- **Architecture:** ray CNN, entity attention, SLAM map (fine + wide views), learned planner, direction
  head → button head, GRU.
- **Training:** DAgger imitation from a training-only teacher, with a self-imitation RL term on a shaped
  reward plus a map-based curiosity bonus. The actor-critic term and the learned curiosity module are in
  the code and in the last run; they are in the weights only if that run beats the submitted agent.
- **Selection rule:** a checkpoint replaces the submission only if it beats the current one on all 180
  held-out maps, and the validator passes.

## Things that did not work (short list)

- PPO fine-tuning, four configurations including one with a trained critic: always lowered the score.
  (Self-imitation RL inside the pooled loop did not.)
- Scan matching for heading: much worse than the action-based turn model.
- Teacher detours for health and shotgun: slightly worse.
- Weight averaging of consecutive rounds: worse than the better round.
- Lower sampling temperature in combat: within noise, and removed as a hand-written condition.

## Rules notes

- The submitted agent's decisions come only from the network. Not learned, and allowed: action masking of
  USE / weapon-switch after a press, observation preprocessing (encoding and the SLAM map), and a crash guard.
- The teacher uses path search and aim maths with privileged state. It lives in `training/` and is never
  imported by `submission/`.
- We do not modify the `main` branch from this workstream; all of this is on `yash`.
