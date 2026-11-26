"""Live viewer: first-person view on the left, the agent's spatial memory on the right.

    python -m training.watch --seed 1000001                 # scripted teacher on a generated level
    python -m training.watch --seed 1000001 --policy agent  # the real submission agent
    python -m training.watch --wad maps/two_rooms.wad

Keys: SPACE pause, RIGHT single-step while paused, +/- speed, ESC quit.
Memory colours: grey = seen free space, white = wall, brown = door, red = locked door,
green tint = already walked, yellow = key seen, magenta = exit seen, cyan dot/line = agent and heading.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from doom_env import make_env
from submission.mapper import DOOR, EXIT, FREE, G, KEY, LOCKED, RES, VISITED, WALL
from training.mapgen import make_level, write_level
from training.scorer import EpisodeScore

VIEW = 480  # pixels per panel
SPAN = 120  # memory cells shown per side


def memory_image(m) -> np.ndarray:
    px, py = (int(v) for v in m.cell(m.x, m.y))
    idx = (np.arange(-SPAN // 2, SPAN // 2) + px) % G
    idy = (np.arange(-SPAN // 2, SPAN // 2) + py) % G
    L = m.layers[:, idx][:, :, idy]  # (layers, x, y)
    img = np.zeros((SPAN, SPAN, 3), np.uint8)
    img[L[FREE] > 0] = (70, 70, 78)
    img[(L[FREE] > 0) & (L[VISITED] > 0)] = (40, 120, 70)
    img[L[WALL] > 0] = (235, 235, 235)
    img[L[DOOR] > 0] = (170, 110, 50)
    img[L[LOCKED] > 0] = (230, 50, 50)
    img[L[KEY] > 0] = (250, 220, 40)
    img[L[EXIT] > 0] = (255, 0, 200)
    return img[:, ::-1].transpose(1, 0, 2)  # x right, y up


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1_000_001)
    ap.add_argument("--tier", type=int, default=None)
    ap.add_argument("--wad", default=None, help="play this WAD instead of a generated level (WAD[:MAP])")
    ap.add_argument("--policy", choices=("teacher", "agent"), default="teacher")
    ap.add_argument("--fps", type=float, default=17.5, help="decisions per second (8.75 = real time)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = play the episode)")
    args = ap.parse_args()

    import pygame

    if args.wad:
        wad, _, map_name = args.wad.partition(":")
        env = make_env(wad, map_name or "MAP01", episode_ticks=4200)
        score = EpisodeScore()
        title = args.wad
    else:
        lvl = make_level(args.seed, args.tier)
        wad = Path(tempfile.gettempdir()) / f"watch_{os.getpid()}.wad"
        write_level(lvl, wad)
        env = make_env(wad, episode_ticks=lvl["ticks"])
        score = EpisodeScore(lvl["monsters"], max(1, lvl["floor_cells"] * lvl["cell"] ** 2 // 128**2))
        title = f"seed {args.seed}  tier {lvl['tier']}"
    obs, _ = env.reset(seed=args.seed)
    score.update(env.privileged_state())
    if args.policy == "teacher":
        from training.teacher import Teacher
        policy = Teacher()
        policy.reset(env)
        mapper = lambda: policy.map
    else:
        from training.play import AgentPolicy
        policy = AgentPolicy()
        policy.reset(env)
        mapper = lambda: getattr(policy.agent, "mapper", None)

    pygame.init()
    screen = pygame.display.set_mode((VIEW * 2 + 12, VIEW + 56))
    pygame.display.set_caption(f"DevForge DOOM viewer - {args.policy} - {title}")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()
    paused, done, steps, fps = False, False, 0, args.fps
    names = ("FWD", "BACK", "LEFT", "RIGHT", "TURN-L", "TURN-R", "FIRE", "USE", "WEAPON")
    action = np.zeros(9, np.int8)
    running = True
    while running:
        step_once = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_RIGHT:
                    step_once = True
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps = min(140.0, fps * 2)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps = max(2.0, fps / 2)
        if not done and (not paused or step_once):
            action = np.asarray(policy.act(obs, env), np.int8)
            obs, _, term, trunc, _ = env.step(action)
            score.update(env.privileged_state())
            steps += 1
            done = term or trunc or (args.max_steps and steps >= args.max_steps)
            if args.max_steps and steps >= args.max_steps:
                running = False
        screen.fill((16, 16, 20))
        frame = pygame.surfarray.make_surface(np.ascontiguousarray(obs["rgb"].transpose(1, 0, 2)))
        screen.blit(pygame.transform.scale(frame, (VIEW, VIEW * 3 // 4)), (0, 0))
        m = mapper()
        if m is not None:
            mem = pygame.surfarray.make_surface(np.ascontiguousarray(memory_image(m).transpose(1, 0, 2)))
            screen.blit(pygame.transform.scale(mem, (VIEW, VIEW)), (VIEW + 12, 0))
            cx, cy = VIEW + 12 + VIEW // 2, VIEW // 2
            pygame.draw.circle(screen, (0, 220, 255), (cx, cy), 5)
            pygame.draw.line(screen, (0, 220, 255), (cx, cy), (cx + 18 * math.cos(m.h), cy - 18 * math.sin(m.h)), 2)
        r = score.result()
        v = obs["vitals"]
        lines = [
            f"step {steps}  hp {v[0]:.0f}  ammo {v[3]:.0f}  keys {int(v[8:14].sum())}   buttons: " + " ".join(n for n, b in zip(names, action) if b),
            f"score {r['score']:.1f}  exit {r['exit']}  doors {r['doors']}  locked {r['locked']}  explored {r['explore']:.0%}  kills {r['kills']}"
            + ("   [PAUSED]" if paused else "") + ("   [EPISODE OVER]" if done else "") + f"   {fps:.0f} steps/s",
        ]
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, (220, 220, 220)), (8, VIEW + 6 + 22 * i))
        pygame.display.flip()
        clock.tick(fps)
    env.close()
    pygame.quit()
    print(score.result())


if __name__ == "__main__":
    main()
