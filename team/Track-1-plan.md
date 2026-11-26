# Track 1 plan: teach first, then let it learn

**The approach:** build a scripted "teacher" bot that cheats with full engine access during training, train the neural network to copy it from the legal observations only, then improve the network with our own reinforcement learning. Pure RL from scratch is unlikely to learn keycard → locked door → exit in hackathon time, whereas copying a teacher gets there in hours. The final agent is fully neural and sees only the official observations, so it stays inside the rules. We say this openly in the brief.

**Deadline:** 00:00 tonight. **Rule:** every stage ends with something submittable, so we are never empty-handed.

On "foolproof": the engineering side (no crashes, no timeouts, valid actions, passes the validator) can be made close to certain. How well the agent plays unseen maps is a training outcome we can measure on held-out maps but cannot guarantee.

## Facts that shape the design

All checked against the template (`docs/`, `env/doom_env/`, `validate.py`, `tests/`).

- **Scoring:** reaching the exit is worth the most, then keycards and doors, then exploration (unique areas), then kills, minus damage and dying. Later tiers weigh more, so navigation and keycard → door → exit matter more than combat. Partial credit counts on every map.
- **Maps are short:** 2,100 to 4,200 ticks at one decision per 4 ticks, so about 525 to 1,050 decisions per map. The agent must move with purpose from the first second.
- **A crash, timeout or invalid action freezes the player** for the rest of that map. Reliability is worth more than cleverness.
- **Limits at evaluation:** 40 ms average per decision, 1 CPU thread, 2 GB VRAM, 8 GB RAM, no network, repo read-only. 60 s for `__init__`, 10 s for `reset()` and the first `act()`, 5 s for any other `act()`.
- **Observations:** 64 rays (distance plus hit type: wall, door, locked door, enemy, item), up to 32 visible entities, 17 vitals (health, armour, ammo, keys, velocity in the player's own frame). **There is no heading and no position**, so any map memory has to track turning from the agent's own actions.
- **Actions:** 9 buttons, each 0 or 1. `USE` only fires on the press, and pressing it on a moving door reverses it, so it must be released between presses.
- **Map format:** the organisers' test maps are UDMF WADs drawn from ASCII grids (`#` wall, `P` player, `D` door, `R`/`B` locked door, `r` key, `E` exit, `z`/`i`/`c` monsters, `h` health). We decoded all 13 fixtures, so our generator can produce maps in exactly that style. The hidden maps probably use it too, but that is an inference.
- **Speed:** about 6,000 environment steps per second on a small grid map and about 1,000 on a full Freedoom level, per core, with pixels turned off.

## The network (about 1 to 2M parameters, about 1 ms per decision)

| Part | Design | Why |
|---|---|---|
| Ray encoder | Circular 1D-CNN over the 64 rays (distance plus hit-type embedding) | Sees walls, doors, locked doors; wraps around 360° |
| Entity encoder | Per-entity MLP, then attention pooling with the mask | Handles any number of enemies and items |
| Vitals encoder | Small MLP | Health, ammo, keycards, velocity |
| Memory v1 | GRU, 256 units | Remembers what it saw |
| Memory v2 (stretch) | Egocentric map of visited and seen space, built from rays plus a heading estimate integrated from our own turn actions, read by a small CNN | Remembers where the locked door was; the innovation story |
| Heads | One output per button, a value head, and helper heads (e.g. "locked door seen?", "direction of the teacher's goal") | Helper heads force the memory to store useful things |

Details:
- **No pixels.** The organisers say they mostly waste training time.
- **Inference always on CPU**, whatever device we are handed. A model this size takes about 1 ms on one thread, and this avoids an untested CUDA path and any risk of hitting the VRAM cap.
- **`USE` debounced by masking:** it is forced to 0 on the step after a press.
- **Sampling at low temperature** instead of always taking the top action, so the agent does not get stuck repeating itself.

## Training pipeline

1. **Map generator.** Grid → UDMF in the organisers' format, thousands of random maps across the four tiers: combat arenas, multi-room mazes with dead ends, keycard → locked door → exit chains, and gauntlets with several locks. Sizes, monster mixes, key colours and door placement are randomised. 15% of seeds are held out and never trained on. Freedoom levels are added as a different-style check.
2. **Local scorer.** Mirrors their categories using `privileged_state()`: exit, keys, doors, sector coverage, kills, damage, death, with the same tick caps. Every checkpoint gets one number on the held-out maps.
3. **Teacher bot (training only).** Reads the real map from the engine, pathfinds to the nearest unexplored area, key, door or exit, and fights enemies on the way. Its score on our held-out maps is our ceiling.
4. **Imitation (DAgger).** The network plays, the teacher labels what it should have done at each step, and the network trains on those labels. Repeating this teaches it to recover from its own mistakes. This is the biggest risk reducer.
5. **RL fine-tune.** Our own recurrent PPO with rewards that mirror the scorer: exit, keycards, doors, new areas, kills, minus damage and death. A fading "stay close to the teacher" term stops it forgetting. Curriculum from tier 1 to tier 4. A checkpoint is kept only if it beats the last one on the local scorer.
6. **Curiosity (stretch).** Random network distillation over the ray and entity encoding as an exploration bonus, replacing a simple visit-count bonus. The organisers name curiosity as heavily rewarded.

## Timeline

Hard stop 03:00 (team deadline, extended from midnight). Times assume work starts at about 16:30.

| Time | Stage | Submittable result |
|---|---|---|
| 16:30–17:00 | Repo layout fixed (template at the repo root), environment ready, random agent passes `validate.py` and is pushed | Random agent |
| 17:00–18:00 | Map generator and local scorer | Random agent with a baseline score |
| 18:00–19:30 | Teacher bot, measured on held-out maps | Random agent |
| 19:30–21:00 | Network plus imitation (DAgger) | **First real neural agent (about 21:00)** |
| 21:00–00:00 | More DAgger rounds, then PPO fine-tune; every checkpoint scored on held-out maps, best one pushed each hour | Best agent so far |
| 00:00–01:30 | Memory v2 and curiosity only if the agent is solid; otherwise more training | Best agent so far |
| 01:30–03:00 | **Feature freeze.** Hardening, final validation, 2-page brief | Final submission |

**Fallback:** if PPO does not beat imitation, we submit the imitation agent. The egocentric map and curiosity are stretch goals; if they don't make it, the brief says so honestly and describes them as next steps.

## Scoring priorities

The automated scorer decides who advances, so every choice is judged by what it earns there:

1. **Never lose a map to a crash, timeout or invalid action.** That forfeits everything after it, so it comes before anything clever.
2. **Exit first.** It is the largest reward and later tiers weigh more, so the generator and the teacher put most training time on navigation and keycard → locked door → exit, not combat.
3. **Explore efficiently.** Unique areas count and spinning earns nothing, so the reward pays for new sectors and the agent is trained to keep moving.
4. **Don't die early, avoid self-damage.** Both are penalised. `SELECT_NEXT_WEAPON` can be masked in situations where it could bring up the rocket launcher at close range (masking from observations is allowed).
5. **Kills last.** They earn points, but the agent should not chase enemies at the cost of progress.
6. **Our local score is the only number we trust.** A checkpoint is pushed only if it beats the last one on held-out maps and Freedoom levels.

## Passing the automated checks

- **Interface:** `DoomAgent` with `__init__` (load weights once), `reset` (zero the GRU and the map), and `act`.
- **Latency:** `torch.set_num_threads(1)`, and a dummy forward pass inside `__init__` so the first `act()` is warm. Time per decision is checked against `limits.py` by the validator, which pins the agent to one core.
- **No crashes:** `act()` wrapped in try/except that returns a safe valid action, outputs clamped to `{0, 1}` with shape `(9,)`, NaN guards on inputs.
- **Sandbox:** no network calls, no writes into the repo, no `vizdoom` import inside the agent.
- **Repo:** created from the template with `submission/agent.py` at the repo root, under 1 GB, weights in Git LFS, pinned `submission/requirements.txt`, all training code committed under `training/`.
- **Code review:** no Stable-Baselines3, and no pathfinding or aim maths in the submitted agent. The teacher lives in `training/`, clearly separate.

## Hardware

- **Training runs on CPU cores:** the game engine is CPU-only, so many game copies run in parallel, one per core. The Apple GPU (MPS) may help with network updates; we measure both and use the faster.
- **MacBook Air M5 (10 cores, 16 GB): the only machine for this version.** Plugged in and kept awake (`caffeinate`) for the night. Training uses 7–8 game copies so the laptop stays usable. It is fanless, so sustained load slows the chip by roughly 30–40%; the timeline allows for that.
- **The ROG is a separate team attempt** (another member's version), not part of this pipeline. The team picks whichever version scores better.
- **Fair comparison:** both versions are scored with the same local scorer on the same held-out maps (`training/` scorer plus a fixed map list), committed to this repo so the other member can run it too.
- **Weights transfer cleanly:** weights trained on a Mac load unchanged on their Linux machine.

## Risks

- **Map style mismatch:** if the hidden maps are not grid-style, generalisation drops. The hedge is mixing Freedoom levels into training and the check set.
- **Heading drift breaks the egocentric map:** the fallback is GRU-only memory.
- **Imitation agent plateaus below the teacher:** helper heads and more DAgger rounds come before PPO.
- **No Docker locally:** the scorer-identical `validate.py --docker` check can't run here. The local validator checks the same limits; only the Linux image differs.
