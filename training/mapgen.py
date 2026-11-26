"""Procedural training maps in the organisers' style: ASCII grid -> UDMF WAD.

The organisers' test maps (tests/fixtures, maps/two_rooms.wad) are UDMF "zdoom" maps drawn from ASCII
grids with 64-unit cells. We write maps the same way so training looks like evaluation, then randomise
layouts across the four tiers (combat, exploration, keycard -> locked door -> exit, gauntlet).

Grid legend
    #  wall            .  floor          P  player start      E  exit pad
    D  door            R/B/Y  locked door (red/blue/yellow)   r/b/y  keycard
    z  zombieman       s  shotgun guy    c  chaingunner        i  imp     d  demon
    h  stimpack        m  medikit        a  clip               S  shells  g  shotgun   A  armor   x  barrel
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from doom_env.wad import Lump, write_wad

CELL = 64
FLOOR_H, CEIL_H = 0, 128

THING_TYPES = {
    "P": 1,
    "z": 3004, "s": 9, "c": 65, "i": 3001, "d": 3002,
    "h": 2011, "m": 2012, "a": 2007, "S": 2008, "g": 2001, "A": 2018, "x": 2035,
    "r": 13, "b": 5, "y": 6,
}
MONSTERS = "zscid"
ITEMS = "hmaSgAx"
KEYS = "rby"
LOCKS = {"R": 129, "B": 130, "Y": 131}  # ZDoom lock numbers: any red / blue / yellow key
KEY_FOR_LOCK = {"R": "r", "B": "b", "Y": "y"}
DOOR_CELLS = "D" + "".join(LOCKS)
DOOR_TEXTURE = {"D": "BIGDOOR2", "R": "DOORRED", "B": "DOORBLU", "Y": "DOORYEL"}


# --------------------------------------------------------------------------------------- grid -> UDMF
def grid_to_udmf(grid: str) -> str:
    rows = grid.strip("\n").split("\n")
    h, w = len(rows), max(len(r) for r in rows)
    rows = [r.ljust(w, "#") for r in rows]

    def at(r, c):
        return rows[r][c] if 0 <= r < h and 0 <= c < w else "#"

    # Sectors: all ordinary floor is sector 0; every connected door / exit component gets its own
    # sector (doors must be separate, or opening one would open them all).
    sector_of: dict[tuple[int, int], int] = {}
    sectors: list[dict] = [{"floor": "FLOOR0_1", "ceil": CEIL_H, "kind": "."}]
    for r in range(h):
        for c in range(w):
            ch = at(r, c)
            if ch == "#" or (r, c) in sector_of:
                continue
            if ch in DOOR_CELLS or ch == "E":
                sid = len(sectors)
                sectors.append({"floor": "FLAT5_4" if ch != "E" else "GATE3", "ceil": FLOOR_H if ch != "E" else CEIL_H, "kind": ch})
                q = deque([(r, c)])
                sector_of[(r, c)] = sid
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        n = (cr + dr, cc + dc)
                        if n not in sector_of and at(*n) == ch:
                            sector_of[n] = sid
                            q.append(n)
            else:
                sector_of[(r, c)] = 0

    vertices: dict[tuple[int, int], int] = {}

    def vid(x, y):
        return vertices.setdefault((x, y), len(vertices))

    sides: list[str] = []
    lines: list[str] = []

    def side(sec, top="-", mid="-"):
        sides.append(f'sidedef {{ sector = {sec}; texturetop = "{top}"; texturemiddle = "{mid}"; texturebottom = "-"; }}')
        return len(sides) - 1

    def corners(r, c):
        x0, x1 = c * CELL, (c + 1) * CELL
        y0, y1 = (h - 1 - r) * CELL, (h - r) * CELL
        return x0, y0, x1, y1

    for (r, c), sa in sector_of.items():
        x0, y0, x1, y1 = corners(r, c)
        ka = sectors[sa]["kind"]
        # Clockwise around the cell => the cell is on the right (front) side of every edge.
        edges = (((x0, y0), (x0, y1), (r, c - 1)), ((x0, y1), (x1, y1), (r - 1, c)),
                 ((x1, y1), (x1, y0), (r, c + 1)), ((x1, y0), (x0, y0), (r + 1, c)))
        for p1, p2, nb in edges:
            sb = sector_of.get(nb)
            if sb is None:  # wall
                mid = "DOORTRAK" if ka in DOOR_CELLS else "STARTAN2"
                lines.append(f"linedef {{ v1 = {vid(*p1)}; v2 = {vid(*p2)}; sidefront = {side(sa, mid=mid)}; blocking = true; dontpegbottom = true; }}")
                continue
            if sb == sa:
                continue
            kb = sectors[sb]["kind"]
            # Emit each two-sided edge once, from the non-door / non-exit side (doors act on the back sector).
            if ka in DOOR_CELLS or ka == "E":
                if kb in DOOR_CELLS or kb == "E":
                    if sa > sb:
                        continue
                else:
                    continue
            special = ""
            if kb in DOOR_CELLS:
                tex = DOOR_TEXTURE[kb]
                if kb == "D":
                    special = "special = 12; arg0 = 0; arg1 = 16; arg2 = 150; playeruse = true; monsteruse = true; repeatspecial = true;"
                else:
                    special = f"special = 13; arg0 = 0; arg1 = 16; arg2 = 0; arg3 = {LOCKS[kb]}; playeruse = true; monsteruse = false; repeatspecial = true;"
                f, b = side(sa, top=tex), side(sb)
            elif kb == "E":
                special = "special = 243; playercross = true;"
                f, b = side(sa), side(sb)
            else:
                f, b = side(sa), side(sb)
            lines.append(f"linedef {{ v1 = {vid(*p1)}; v2 = {vid(*p2)}; sidefront = {f}; sideback = {b}; twosided = true; {special} }}")

    things = []
    for r in range(h):
        for c in range(w):
            ch = at(r, c)
            if ch in THING_TYPES:
                x0, y0, _, _ = corners(r, c)
                angle = 0 if ch == "P" else 180
                things.append(f"thing {{ x = {x0 + CELL / 2:.1f}; y = {y0 + CELL / 2:.1f}; angle = {angle}; type = {THING_TYPES[ch]}; "
                              "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }")

    out = ['namespace = "zdoom";']
    out += [f"vertex {{ x = {x:.1f}; y = {y:.1f}; }}" for (x, y), _ in sorted(vertices.items(), key=lambda kv: kv[1])]
    out += [f'sector {{ texturefloor = "{s["floor"]}"; textureceiling = "CEIL1_1"; heightfloor = {FLOOR_H}; heightceiling = {s["ceil"]}; lightlevel = 192; }}' for s in sectors]
    out += sides + lines + things
    return "\n".join(out) + "\n"


def write_map(grid: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", grid_to_udmf(grid).encode()), Lump("ENDMAP", b"")])
    return path


# --------------------------------------------------------------------------------------- layouts
@dataclass
class Room:
    r0: int
    c0: int
    r1: int  # inclusive
    c1: int

    def cells(self):
        return [(r, c) for r in range(self.r0, self.r1 + 1) for c in range(self.c0, self.c1 + 1)]

    def center(self):
        return ((self.r0 + self.r1) // 2, (self.c0 + self.c1) // 2)

    def overlaps(self, o: "Room", margin: int = 1) -> bool:
        return not (self.r1 + margin < o.r0 or o.r1 + margin < self.r0 or self.c1 + margin < o.c0 or o.c1 + margin < self.c0)


@dataclass
class Layout:
    grid: list[list[str]]
    rooms: list[Room]
    edges: list[tuple[int, int, list[tuple[int, int]]]] = field(default_factory=list)  # (room a, room b, corridor path)

    def text(self) -> str:
        return "\n".join("".join(row) for row in self.grid)


TIER_TICKS = {1: 2100, 2: 2800, 3: 3500, 4: 4200}


def _carve_rooms(rng: random.Random, h: int, w: int, n_rooms: int, size=(3, 8)) -> Layout:
    grid = [["#"] * w for _ in range(h)]
    rooms: list[Room] = []
    for _ in range(n_rooms * 30):
        if len(rooms) >= n_rooms:
            break
        rh, rw = rng.randint(*size), rng.randint(*size)
        r0, c0 = rng.randint(1, h - rh - 1), rng.randint(1, w - rw - 1)
        room = Room(r0, c0, r0 + rh - 1, c0 + rw - 1)
        if any(room.overlaps(o, margin=2) for o in rooms):
            continue
        rooms.append(room)
        for r, c in room.cells():
            grid[r][c] = "."
    return Layout(grid, rooms)


def _corridor(rng: random.Random, a: Room, b: Room) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = (rng.randint(a.r0, a.r1), rng.randint(a.c0, a.c1)), (rng.randint(b.r0, b.r1), rng.randint(b.c0, b.c1))
    path = []
    if rng.random() < 0.5:
        for c in range(c0, c1 + (1 if c1 >= c0 else -1), 1 if c1 >= c0 else -1):
            path.append((r0, c))
        for r in range(r0, r1 + (1 if r1 >= r0 else -1), 1 if r1 >= r0 else -1):
            path.append((r, c1))
    else:
        for r in range(r0, r1 + (1 if r1 >= r0 else -1), 1 if r1 >= r0 else -1):
            path.append((r, c0))
        for c in range(c0, c1 + (1 if c1 >= c0 else -1), 1 if c1 >= c0 else -1):
            path.append((r1, c))
    out = []
    for p in path:
        if not out or out[-1] != p:
            out.append(p)
    return out


def _connect(rng: random.Random, lay: Layout, extra: float) -> None:
    n = len(lay.rooms)
    centers = [r.center() for r in lay.rooms]
    in_tree, edges = {0}, []
    while len(in_tree) < n:  # Prim's MST on Manhattan distance between room centres
        best = min(((abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1]), i, j)
                    for i in in_tree for j in range(n) if j not in in_tree))
        edges.append((best[1], best[2]))
        in_tree.add(best[2])
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edges and (j, i) not in edges and rng.random() < extra / n:
                edges.append((i, j))
    for i, j in edges:
        path = _corridor(rng, lay.rooms[i], lay.rooms[j])
        for r, c in path:
            lay.grid[r][c] = "."
        lay.edges.append((i, j, path))


def _room_of(lay: Layout, cell) -> int | None:
    r, c = cell
    for k, room in enumerate(lay.rooms):
        if room.r0 <= r <= room.r1 and room.c0 <= c <= room.c1:
            return k
    return None


def _door_spot(lay: Layout, path, near_end: bool) -> tuple[int, int] | None:
    """A corridor cell outside every room with walls on both sides across the corridor."""
    g = lay.grid
    cells = list(reversed(path)) if near_end else list(path)
    for r, c in cells:
        if _room_of(lay, (r, c)) is not None or g[r][c] != ".":
            continue
        # Doors must not touch another door: a Doom door only rises to its lowest neighbouring
        # ceiling, so a door next to a closed door can never open.
        if any(g[r + dr][c + dc] in DOOR_CELLS for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))):
            continue
        ns = g[r - 1][c] == "#" and g[r + 1][c] == "#" and g[r][c - 1] != "#" and g[r][c + 1] != "#"
        ew = g[r][c - 1] == "#" and g[r][c + 1] == "#" and g[r - 1][c] != "#" and g[r + 1][c] != "#"
        if ns or ew:
            return (r, c)
    return None


def _reachable(grid, start, keys: set[str]) -> set:
    h, w = len(grid), len(grid[0])
    seen, q = {start}, deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in seen:
                continue
            ch = grid[nr][nc]
            if ch == "#" or (ch in LOCKS and KEY_FOR_LOCK[ch] not in keys):
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def solvable(grid, start) -> tuple[bool, list[str]]:
    """Simulate key collection. Returns (exit reachable, keys in pickup order)."""
    keys: list[str] = []
    while True:
        reach = _reachable(grid, start, set(keys))
        if any(grid[r][c] == "E" for r, c in reach):
            return True, keys
        new = sorted({grid[r][c] for r, c in reach if grid[r][c] in KEYS} - set(keys))
        if not new:
            return False, keys
        keys += new


def _free_cells(lay: Layout, room: Room, avoid: set) -> list:
    return [(r, c) for r, c in room.cells() if lay.grid[r][c] == "." and (r, c) not in avoid]


def generate(tier: int, seed: int) -> tuple[str, int]:
    """Returns (grid text, episode ticks) for a random map of the given tier."""
    rng = random.Random(seed * 7919 + tier)
    for _ in range(200):
        out = _try_generate(rng, tier)
        if out is not None:
            return out, min(4200, TIER_TICKS[tier] + rng.choice((0, 0, 350, 700)) * (tier > 1))
    raise RuntimeError(f"could not generate tier {tier} seed {seed}")


def _try_generate(rng: random.Random, tier: int) -> str | None:
    if tier == 1:
        h, w, n_rooms, size = rng.randint(12, 20), rng.randint(14, 24), rng.randint(1, 3), (5, 10)
    elif tier == 2:
        h, w, n_rooms, size = rng.randint(22, 34), rng.randint(22, 34), rng.randint(4, 8), (3, 7)
    elif tier == 3:
        h, w, n_rooms, size = rng.randint(22, 34), rng.randint(22, 34), rng.randint(4, 7), (3, 7)
    else:
        h, w, n_rooms, size = rng.randint(28, 40), rng.randint(28, 40), rng.randint(6, 9), (3, 7)
    lay = _carve_rooms(rng, h, w, n_rooms, size)
    if len(lay.rooms) < max(1, n_rooms - 2):
        return None
    if len(lay.rooms) > 1:
        _connect(rng, lay, extra=0.0 if tier >= 3 else rng.choice((0.0, 1.0, 2.0)))
    g = lay.grid
    n = len(lay.rooms)

    # Pillars in big combat rooms for cover.
    for room in lay.rooms:
        if (room.r1 - room.r0) >= 5 and (room.c1 - room.c0) >= 5 and rng.random() < 0.5:
            for _ in range(rng.randint(1, 3)):
                r, c = rng.randint(room.r0 + 2, room.r1 - 2), rng.randint(room.c0 + 2, room.c1 - 2)
                g[r][c] = "#"

    start_room = 0
    used: set = set()
    start = rng.choice(_free_cells(lay, lay.rooms[start_room], used))
    used.add(start)

    # Room graph distances (via corridor edges) from the start room.
    adj = {i: [] for i in range(n)}
    for i, j, _ in lay.edges:
        adj[i].append(j)
        adj[j].append(i)
    dist = {start_room: 0}
    q = deque([start_room])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    leaves = [i for i in range(n) if i != start_room and len(adj[i]) == 1]

    # Plain doors on some corridors.
    p_door = {1: 0.3, 2: 0.6, 3: 0.5, 4: 0.6}[tier]
    for i, j, path in lay.edges:
        if rng.random() < p_door:
            spot = _door_spot(lay, path, near_end=rng.random() < 0.5)
            if spot:
                g[spot[0]][spot[1]] = "D"

    # Exit.
    has_exit = tier > 1 or rng.random() < 0.5
    exit_room = None
    if has_exit:
        if n == 1:
            exit_room = 0
        else:
            cands = leaves if tier >= 3 and leaves else [i for i in range(n) if i != start_room]
            far = max(dist.get(i, 0) for i in cands)
            exit_room = rng.choice([i for i in cands if dist.get(i, 0) >= far - (1 if tier == 2 else 0)])
        cells = [p for p in _free_cells(lay, lay.rooms[exit_room], used) if abs(p[0] - start[0]) + abs(p[1] - start[1]) > 3]
        if not cells:
            return None
        e = rng.choice(cells)
        g[e[0]][e[1]] = "E"
        used.add(e)

    # Locks: tier 3 one lock on the exit room's corridor, tier 4 a chain of two.
    colors = rng.sample(["R", "B", "Y"], 3)
    if tier >= 3:
        if exit_room is None or exit_room == start_room:
            return None
        chain = [exit_room]
        if tier == 4:
            others = [i for i in leaves if i not in (exit_room, start_room)]
            if not others:
                return None
            chain.append(rng.choice(others))
        for k, room_id in enumerate(chain):
            edge = next(((i, j, p) for i, j, p in lay.edges if room_id in (i, j)), None)
            if edge is None:
                return None
            i, j, path = edge
            spot = _door_spot(lay, path, near_end=(j == room_id))
            if spot is None:
                return None
            g[spot[0]][spot[1]] = colors[k]
        # Key for lock k goes in the room behind lock k+1 (or anywhere reachable for the last one).
        for k, room_id in enumerate(chain):
            key = KEY_FOR_LOCK[colors[k]]
            if k + 1 < len(chain):
                cells = _free_cells(lay, lay.rooms[chain[k + 1]], used)
            else:
                reach = _reachable(g, start, set())
                pool = [i for i in range(n) if i not in chain and i != start_room] or [start_room]
                room_k = max(pool, key=lambda i: dist.get(i, 0) + rng.random() * 2)
                cells = [p for p in _free_cells(lay, lay.rooms[room_k], used) if p in reach]
            if not cells:
                return None
            kc = rng.choice(cells)
            g[kc[0]][kc[1]] = key
            used.add(kc)

    # Remove plain doors that ended up touching a locked door (see _door_spot).
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if g[r][c] == "D" and any(g[r + dr][c + dc] in LOCKS for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                g[r][c] = "."

    # Monsters and items.
    n_mon = {1: rng.randint(3, 9), 2: rng.randint(2, 7), 3: rng.randint(2, 7), 4: rng.randint(6, 14)}[tier]
    weights = {"z": 5, "s": 3, "i": 5, "c": 1, "d": 2}
    mon_pool = "".join(k * v for k, v in weights.items())
    room_cells = [p for room in lay.rooms for p in _free_cells(lay, room, used)]
    far_cells = [p for p in room_cells if abs(p[0] - start[0]) + abs(p[1] - start[1]) >= 5]
    rng.shuffle(far_cells)
    for p in far_cells[:n_mon]:
        g[p[0]][p[1]] = rng.choice(mon_pool)
        used.add(p)
    n_items = {1: rng.randint(2, 6), 2: rng.randint(2, 6), 3: rng.randint(2, 6), 4: rng.randint(1, 4)}[tier]
    free = [p for p in room_cells if p not in used]
    rng.shuffle(free)
    for p in free[:n_items]:
        g[p[0]][p[1]] = rng.choice("hhmaaSSgAx")
        used.add(p)
    g[start[0]][start[1]] = "P"

    if has_exit:
        ok, keys = solvable(g, start)
        if not ok:
            return None
        if tier >= 3:
            if any(g[r][c] == "E" for r, c in _reachable(g, start, set())):
                return None  # exit must really be behind the lock
            if len(keys) < tier - 2:
                return None
    return "\n".join("".join(row) for row in g)


def generate_set(out_dir: str | Path, tiers=(1, 2, 3, 4), per_tier: int = 100, seed0: int = 0) -> list[dict]:
    """Write per_tier maps for each tier. Returns [{path, tier, seed, ticks}]."""
    out_dir = Path(out_dir)
    items = []
    for tier in tiers:
        for k in range(per_tier):
            seed = seed0 + k
            grid, ticks = generate(tier, seed)
            path = write_map(grid, out_dir / f"t{tier}_{seed:05d}.wad")
            (out_dir / f"t{tier}_{seed:05d}.txt").write_text(grid + "\n")
            items.append({"path": str(path), "tier": tier, "seed": seed, "ticks": ticks})
    return items


if __name__ == "__main__":
    import sys

    tier = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    grid, ticks = generate(tier, seed)
    print(grid)
    print("ticks", ticks)
