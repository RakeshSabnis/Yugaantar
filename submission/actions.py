"""Discrete action set: the policy picks one button combination per step.

Button order follows docs/CONTRACT.md §3:
0 MOVE_FORWARD, 1 MOVE_BACKWARD, 2 MOVE_LEFT, 3 MOVE_RIGHT, 4 TURN_LEFT, 5 TURN_RIGHT, 6 ATTACK, 7 USE, 8 SELECT_NEXT_WEAPON
"""

from __future__ import annotations

import numpy as np

F, B, SL, SR, TL, TR, ATK, USE, NEXTW = range(9)

COMBOS: tuple[tuple[int, ...], ...] = (
    (),             # 0 no-op
    (F,),           # 1
    (F, TL),        # 2
    (F, TR),        # 3
    (TL,),          # 4
    (TR,),          # 5
    (B,),           # 6
    (SL,),          # 7
    (SR,),          # 8
    (F, SL),        # 9
    (F, SR),        # 10
    (ATK,),         # 11
    (ATK, TL),      # 12
    (ATK, TR),      # 13
    (ATK, SL),      # 14
    (ATK, SR),      # 15
    (F, ATK),       # 16
    (USE,),         # 17
    (F, USE),       # 18
    (NEXTW,),       # 19
    (B, TL),        # 20
    (B, TR),        # 21
    (B, ATK),       # 22
)
NUM_ACTIONS = len(COMBOS)
USE_ACTIONS = tuple(i for i, c in enumerate(COMBOS) if USE in c)
NEXTW_ACTIONS = tuple(i for i, c in enumerate(COMBOS) if NEXTW in c)

_TABLE = np.zeros((NUM_ACTIONS, 9), np.int8)
for _i, _c in enumerate(COMBOS):
    _TABLE[_i, list(_c)] = 1


def to_buttons(index: int) -> np.ndarray:
    return _TABLE[int(index)].copy()


def from_buttons(buttons) -> int:
    """Nearest combo for a button vector (exact match if present)."""
    b = np.asarray(buttons, np.int8).reshape(9)
    return int(np.argmin(np.abs(_TABLE - b).sum(axis=1)))
