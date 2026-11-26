"""Procedural training maps: ASCII grid -> UDMF WAD, plus random level generators for the four tiers.

The WAD writer mirrors the format of the organizers' own fixtures (tests/fixtures/*.wad): 64-unit
cells, one sector per connected region, Door_Raise (12) / Door_LockedRaise (13) door cells and an
Exit_Normal (243) walk-over exit cell. Evaluation maps are hidden, so everything else is randomized
(room sizes, corridor widths, cell scale, monster mix, key colours, start angle) to avoid overfitting
to one style.

Legend
    #  wall            .  floor           P  player start       E  exit cell
    D  door            R/B/Y locked door (red/blue/yellow)      r/b/y  key card
    z  zombieman   s  shotgun guy   c  chaingunner   i  imp   d  demon   l  lost soul   k  hell knight
    h  stimpack    m  medikit       a  ammo clip     x  shell box   g  shotgun   v  green armor

CLI:  python -m training.mapgen --out data/maps --count 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from doom_env.wad import Lump, write_wad

MONSTERS = {"z": 3004, "s": 9, "c": 65, "i": 3001, "d": 3002, "l": 3006, "k": 69}
ITEMS = {"h": 2011, "m": 2012, "a": 2007, "x": 2049, "g": 2001, "v": 2018}
KEYS = {"r": 13, "b": 5, "y": 6}
LOCKS = {"R": 129, "B": 130, "Y": 131}  # ZDoom lock numbers: any red / blue / yellow key
LOCK_OF_KEY = {"r": "R", "b": "B", "y": "Y"}
DOOR_TEX = {"D": "BIGDOOR2", "R": "DOORRED", "B": "DOORBLU", "Y": "DOORYEL"}
FLOORS = ("FLOOR0_1", "FLOOR4_8", "FLAT14", "FLOOR5_1", "FLAT1", "FLOOR0_3")
WALLS = ("STARTAN2", "STONE2", "BROWN1", "GRAY1", "BRICK1", "METAL1")
DOORS = "DRBY"


# ----------------------------------------------------------------------------- grid -> UDMF
def parse(grid: str) -> list[list[str]]:
    rows = [list(r) for r in grid.strip("\n").split("\n")]
    w = max(len(r) for r in rows)
    return [r + ["#"] * (w - len(r)) for r in rows]


def _regions(g):
    """Sector id per cell (-1 = wall). Connected cells of the same class share a sector."""
    h, w = len(g), len(g[0])
    cls = lambda ch: None if ch == "#" else (ch if ch in DOORS or ch == "E" else ".")
    sec = [[-1] * w for _ in range(h)]
    kinds: list[str] = []
    for r in range(h):
        for c in range(w):
            k = cls(g[r][c])
            if k is None or sec[r][c] >= 0:
                continue
            sid = len(kinds)
            kinds.append(k)
            todo = [(r, c)]
            sec[r][c] = sid
            while todo:
                y, x = todo.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and sec[ny][nx] < 0 and cls(g[ny][nx]) == k:
                        sec[ny][nx] = sid
                        todo.append((ny, nx))
    return sec, kinds


def grid_to_textmap(grid: str, cell: int = 64, rng: np.random.Generator | None = None, start_angle: int | None = 0) -> str:
    g = parse(grid)
    h, w = len(g), len(g[0])
    sec, kinds = _regions(g)
    S = lambda r, c: sec[r][c] if 0 <= r < h and 0 <= c < w else -1
    # Unit edges keyed by (axis, fixed coordinate, direction, front sector, back sector) -> positions.
    # The front sector lies on the right-hand side of v1 -> v2 (Doom convention).
    edges: dict[tuple, list[int]] = {}
    for r in range(h):
        for c in range(w):
            s = sec[r][c]
            if s < 0:
                continue
            y0, x0 = (h - 1 - r), c  # cell origin in cell units, y up
            for (dr, dc, axis, fixed, pos, direction) in (
                (0, 1, "v", x0 + 1, y0, -1), (0, -1, "v", x0, y0, +1),
                (-1, 0, "h", y0 + 1, x0, +1), (1, 0, "h", y0, x0, -1),
            ):
                t = S(r + dr, c + dc)
                if t == s:
                    continue
                if t >= 0:
                    # Each two-sided line is emitted once. A tag-0 door special acts on the line's
                    # BACK sector (and doom_env reads it the same way), so the door must be the back.
                    sd, td = kinds[s] in DOORS, kinds[t] in DOORS
                    if (sd and not td) or (sd == td and t < s):
                        continue
                edges.setdefault((axis, fixed, direction, s, t), []).append(pos)

    verts: dict[tuple, int] = {}
    def V(x, y):
        return verts.setdefault((float(x * cell), float(y * cell)), len(verts))

    wall_tex = [WALLS[(rng.integers(len(WALLS)) if rng is not None else i) % len(WALLS)] for i in range(len(kinds))]
    sides: list[str] = []
    lines: list[str] = []
    def side(sector, top="-", mid="-"):
        sides.append(f'sidedef {{ sector = {sector}; texturetop = "{top}"; texturemiddle = "{mid}"; texturebottom = "-"; }}')
        return len(sides) - 1

    for (axis, fixed, direction, s, t), poss in sorted(edges.items()):
        poss.sort()
        runs, start, prev = [], poss[0], poss[0]
        for p in poss[1:]:
            if p != prev + 1:
                runs.append((start, prev + 1))
                start = p
            prev = p
        runs.append((start, prev + 1))
        for a, b in runs:
            lo, hi = (a, b) if direction > 0 else (b, a)
            v1, v2 = (V(fixed, lo), V(fixed, hi)) if axis == "v" else (V(lo, fixed), V(hi, fixed))
            if t < 0:
                mid = "DOORTRAK" if kinds[s] in DOORS else wall_tex[s]
                lines.append(f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {side(s, mid=mid)}; blocking = true; dontpegbottom = true; }}")
                continue
            ks, kt = kinds[s], kinds[t]
            door = ks if ks in DOORS else (kt if kt in DOORS else None)
            if door:
                tex = DOOR_TEX[door]
                f = side(s, top="-" if ks in DOORS else tex)
                bk = side(t, top="-" if kt in DOORS else tex)
                if door == "D":
                    spec = "special = 12; arg0 = 0; arg1 = 16; arg2 = 150; playeruse = true; monsteruse = true; repeatspecial = true;"
                else:
                    spec = f"special = 13; arg0 = 0; arg1 = 16; arg2 = 0; arg3 = {LOCKS[door]}; playeruse = true; monsteruse = false; repeatspecial = true;"
            else:
                f, bk = side(s), side(t)
                spec = "special = 243; playercross = true;" if "E" in (ks, kt) else ""
            lines.append(f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {f}; sideback = {bk}; twosided = true; {spec} }}")

    out = ['namespace = "zdoom";']
    out += [f"vertex {{ x = {x}; y = {y}; }}" for (x, y), _ in sorted(verts.items(), key=lambda kv: kv[1])]
    for i, k in enumerate(kinds):
        floor = "FLAT5_4" if k in DOORS else "GATE3" if k == "E" else FLOORS[i % len(FLOORS)]
        out.append(f'sector {{ texturefloor = "{floor}"; textureceiling = "CEIL1_1"; heightfloor = 0; '
                   f"heightceiling = {0 if k in DOORS else 128}; lightlevel = 192; }}")
    out += sides + lines
    flags = "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true;"
    for r in range(h):
        for c in range(w):
            ch = g[r][c]
            typ = 1 if ch == "P" else MONSTERS.get(ch) or ITEMS.get(ch) or KEYS.get(ch)
            if not typ:
                continue
            if ch == "P":
                ang = start_angle if start_angle is not None else int(rng.integers(8)) * 45
            else:
                ang = 180 if rng is None else int(rng.integers(8)) * 45
            x, y = (c + 0.5) * cell, (h - 1 - r + 0.5) * cell
            out.append(f"thing {{ x = {x}; y = {y}; angle = {ang}; type = {typ}; {flags} }}")
    return "\n".join(out) + "\n"


def write_map(grid: str, path: str | Path, cell: int = 64, rng=None, start_angle: int | None = 0) -> None:
    text = grid_to_textmap(grid, cell, rng, start_angle)
    write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode()), Lump("ENDMAP", b"")])


# ----------------------------------------------------------------------------- solver
def solve(g) -> int | None:
    """Shortest number of cell moves from P to an E cell collecting keys as needed (BFS over
    (cell, keys)); None if unsolvable. Used to reject broken layouts and to size the tick cap."""
    h, w = len(g), len(g[0])
    start = next(((r, c) for r in range(h) for c in range(w) if g[r][c] == "P"), None)
    if start is None:
        return None
    bit = {"r": 1, "b": 2, "y": 4, "R": 1, "B": 2, "Y": 4}
    seen = {(start, 0)}
    todo = deque([(start, 0, 0)])
    while todo:
        (r, c), keys, d = todo.popleft()
        if g[r][c] == "E":
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or g[nr][nc] == "#":
                continue
            ch = g[nr][nc]
            if ch in "RBY" and not keys & bit[ch]:
                continue
            nk = keys | bit[ch] if ch in "rby" else keys
            if ((nr, nc), nk) not in seen:
                seen.add(((nr, nc), nk))
                todo.append(((nr, nc), nk, d + 1))
    return None


# ----------------------------------------------------------------------------- random levels
TIER_CFG = {
    #       rooms     locks  monsters/room  monster pool                    door prob  loops
    1: dict(rooms=(1, 3), locks=0, mons=(1.5, 4.0), pool="zzzzsiiid", door=0.3, loops=1),
    2: dict(rooms=(5, 9), locks=0, mons=(0.3, 1.0), pool="zzzsii", door=0.6, loops=2),
    3: dict(rooms=(5, 9), locks=1, mons=(0.4, 1.2), pool="zzzsiid", door=0.7, loops=1),
    4: dict(rooms=(7, 12), locks=(2, 3), mons=(0.6, 1.6), pool="zzzssiiidc", door=0.7, loops=2),
}


def _carve_corridor(g, a, b, rng, wide: bool):
    """L-shaped corridor between cells a and b. Returns the carved cells in order."""
    (r0, c0), (r1, c1) = a, b
    path = []
    horizontal_first = rng.random() < 0.5
    r, c = r0, c0
    def walk_c():
        nonlocal c
        while c != c1:
            c += 1 if c1 > c else -1
            path.append((r, c))
    def walk_r():
        nonlocal r
        while r != r1:
            r += 1 if r1 > r else -1
            path.append((r, c))
    (walk_c(), walk_r()) if horizontal_first else (walk_r(), walk_c())
    return path


def generate(rng: np.random.Generator, tier: int) -> dict | None:
    """One random level. Returns {"grid", "tier", "ticks", "cell", ...} or None if the attempt failed."""
    cfg = TIER_CFG[tier]
    n_rooms = int(rng.integers(cfg["rooms"][0], cfg["rooms"][1] + 1))
    n_locks = cfg["locks"] if isinstance(cfg["locks"], int) else int(rng.integers(cfg["locks"][0], cfg["locks"][1] + 1))
    big = tier == 1
    size = int(14 + 5.5 * np.sqrt(n_rooms) + rng.integers(0, 6))
    H = W = size
    g = [["#"] * W for _ in range(H)]
    rooms = []
    for _ in range(400):
        if len(rooms) == n_rooms:
            break
        rh, rw = (int(rng.integers(5, 11)), int(rng.integers(5, 11))) if big else (int(rng.integers(2, 7)), int(rng.integers(2, 7)))
        r0, c0 = int(rng.integers(1, max(2, H - rh - 1))), int(rng.integers(1, max(2, W - rw - 1)))
        if r0 + rh >= H - 1 or c0 + rw >= W - 1:
            continue
        if any(r0 - 2 < R + RH and R - 2 < r0 + rh and c0 - 2 < C + RW and C - 2 < c0 + rw for R, C, RH, RW in rooms):
            continue
        rooms.append((r0, c0, rh, rw))
    if len(rooms) < max(1, cfg["rooms"][0]):
        return None
    n = len(rooms)
    owner = {}
    for i, (r0, c0, rh, rw) in enumerate(rooms):
        for r in range(r0, r0 + rh):
            for c in range(c0, c0 + rw):
                g[r][c] = "."
                owner[(r, c)] = i
        if rh >= 5 and rw >= 5 and rng.random() < 0.5:  # pillars: cover and non-convex rooms
            for _ in range(int(rng.integers(1, 3))):
                pr, pc = int(rng.integers(r0 + 1, r0 + rh - 1)), int(rng.integers(c0 + 1, c0 + rw - 1))
                g[pr][pc] = "#"
                owner.pop((pr, pc), None)
    centre = lambda i: (rooms[i][0] + rooms[i][2] // 2, rooms[i][1] + rooms[i][3] // 2)
    for i in range(n):  # pillars must not land on the cell corridors start from
        r, c = centre(i)
        g[r][c] = "."
        owner[(r, c)] = i
    dist = lambda i, j: abs(centre(i)[0] - centre(j)[0]) + abs(centre(i)[1] - centre(j)[1])

    # Spanning tree (Prim on centre distance) from a random start room.
    start = int(rng.integers(n))
    parent = {start: None}
    while len(parent) < n:
        i, j = min(((i, j) for i in parent for j in range(n) if j not in parent), key=lambda e: dist(*e) + rng.random() * 4)
        parent[j] = i
    depth = {}
    def dep(i):
        if i not in depth:
            depth[i] = 0 if parent[i] is None else dep(parent[i]) + 1
        return depth[i]
    exit_room = max(range(n), key=lambda i: (dep(i), rng.random()))
    main = []
    i = exit_room
    while i is not None:
        main.append(i)
        i = parent[i]
    main.reverse()  # start ... exit
    if n_locks and len(main) < n_locks + 1:
        return None
    lock_edges = {}
    if n_locks:
        idx = sorted(rng.choice(np.arange(1, len(main)), size=n_locks, replace=False).tolist())
        colours = list(rng.permutation(list("rby")))[:n_locks]
        lock_edges = {main[k]: col for k, col in zip(idx, colours)}  # child room -> key colour of the edge into it
    zone = {}
    def zn(i):
        if i not in zone:
            zone[i] = 0 if parent[i] is None else zn(parent[i]) + (1 if i in lock_edges else 0)
        return zone[i]
    for i in range(n):
        zn(i)

    edges = [(parent[j], j) for j in parent if parent[j] is not None]
    extra = [(i, j) for i in range(n) for j in range(i + 1, n)
             if zone[i] == zone[j] and parent[j] != i and parent[i] != j and dist(i, j) < size * 0.6]
    rng.shuffle(extra)
    edges += extra[: int(rng.integers(0, cfg["loops"] + 1))]

    door_cells = {}
    for (i, j) in edges:
        path = _carve_corridor(g, centre(i), centre(j), rng, False)
        outside = [p for p in path if p not in owner]
        for p in outside:
            if g[p[0]][p[1]] == "#":
                g[p[0]][p[1]] = "."
        if any(owner.get(p) not in (None, i, j) for p in path):
            if parent.get(j) == i and j in lock_edges:
                return None  # the locked corridor cuts through another room: the lock could be bypassed
            continue
        locked = parent.get(j) == i and j in lock_edges
        if outside and (locked or rng.random() < cfg["door"]):
            # the door goes at the end of the corridor next to the child room (or at a random spot)
            p = outside[-1] if locked or rng.random() < 0.6 else outside[int(rng.integers(len(outside)))]
            door_cells[p] = LOCK_OF_KEY[lock_edges[j]] if locked else "D"
        elif locked:
            return None
    for (r, c), ch in door_cells.items():
        open_ns = g[r - 1][c] != "#" and g[r + 1][c] != "#"
        open_ew = g[r][c - 1] != "#" and g[r][c + 1] != "#"
        wall_ns = g[r - 1][c] == "#" and g[r + 1][c] == "#"
        wall_ew = g[r][c - 1] == "#" and g[r][c + 1] == "#"
        if (open_ns and wall_ew) or (open_ew and wall_ns):
            g[r][c] = ch
        elif ch != "D":
            return None  # a locked door that can't seal its corridor
    # doors must not touch each other or the exit
    for (r, c) in list(door_cells):
        if g[r][c] in DOORS and any(g[r + dr][c + dc] in DOORS for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            if g[r][c] != "D":
                return None
            g[r][c] = "."

    free = lambda i: [(r, c) for (r, c), o in owner.items() if o == i and g[r][c] == "."]
    def place(i, ch, far_from=None):
        cells = free(i)
        if not cells:
            return False
        if far_from is not None:
            cells.sort(key=lambda p: -(abs(p[0] - far_from[0]) + abs(p[1] - far_from[1])))
            cells = cells[: max(1, len(cells) // 3)]
        r, c = cells[int(rng.integers(len(cells)))]
        g[r][c] = ch
        return True

    pr, pc = centre(start)
    g[pr][pc] = "P"
    # exit: a cell of the exit room far from its entrance, not next to a door
    cand = [p for p in free(exit_room) if not any(g[p[0] + dr][p[1] + dc] in DOORS for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    if not cand:
        return None
    ent = centre(parent[exit_room]) if parent[exit_room] is not None else (pr, pc)
    cand.sort(key=lambda p: -(abs(p[0] - ent[0]) + abs(p[1] - ent[1])))
    er, ec = cand[int(rng.integers(max(1, len(cand) // 4)))]
    g[er][ec] = "E"
    # keys: in a room of the zone just before the lock, preferring rooms off the main path
    for child, col in lock_edges.items():
        z = zone[child] - 1
        rooms_z = [i for i in range(n) if zone[i] == z]
        off = [i for i in rooms_z if i not in main] or rooms_z
        if not place(off[int(rng.integers(len(off)))], col, far_from=centre(parent[child])):
            return None
    # monsters and pickups
    lo, hi = cfg["mons"]
    n_mon = 0
    for i in range(n):
        k = int(rng.poisson(rng.uniform(lo, hi)))
        if i == start and tier > 1:
            k = min(k, 1)
        for _ in range(k):
            n_mon += place(i, cfg["pool"][int(rng.integers(len(cfg["pool"])))], far_from=(pr, pc) if i == start else None)
        if rng.random() < (0.5 if tier in (1, 4) else 0.25):
            place(i, "hhmaaxgv"[int(rng.integers(8))])
    if tier == 1 and n_mon == 0:
        return None
    steps = solve(g)
    if steps is None:
        return None
    cell = int(rng.choice([64, 64, 64, 96, 128]))
    need = steps * cell / 8.0 * 4 + 350 * (n_locks + 1) + 60 * n_mon  # generous: ~8 units/tic walking
    ticks = int(np.clip(np.ceil(need * 1.6 / 350) * 350, 2100, 4200))
    grid = "\n".join("".join(r) for r in g)
    return {"grid": grid, "tier": tier, "ticks": ticks, "cell": cell, "path_cells": steps, "monsters": n_mon,
            "locks": n_locks, "floor_cells": sum(ch != "#" for r in g for ch in r)}


def make_level(seed: int, tier: int | None = None) -> dict:
    """Deterministic level for a seed (tier cycles 1..4 with the seed unless given)."""
    tier = tier or (seed % 4) + 1
    rng = np.random.default_rng([seed, tier])
    while True:
        lvl = generate(rng, tier)
        if lvl is not None:
            lvl["seed"] = seed
            lvl["start_angle"] = int(rng.integers(8)) * 45
            return lvl


def write_level(lvl: dict, path: str | Path) -> None:
    write_map(lvl["grid"], path, lvl["cell"], np.random.default_rng(lvl["seed"]), lvl["start_angle"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/maps")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="first seed; seeds >= 1_000_000 are reserved for held-out evaluation")
    ap.add_argument("--tier", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for s in range(args.seed, args.seed + args.count):
        lvl = make_level(s, args.tier)
        name = f"t{lvl['tier']}_{s:07d}.wad"
        write_level(lvl, out / name)
        index.append({"wad": name, **{k: v for k, v in lvl.items() if k != "grid"}, "grid": lvl["grid"]})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} maps to {out}")


if __name__ == "__main__":
    main()
