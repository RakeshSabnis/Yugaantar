"""Generate the training and held-out map sets.

    uv run python training/make_maps.py            # maps_gen/train (seeds 0..) and maps_gen/heldout (seeds 100000..)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mapgen import generate_set  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    per_tier = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    held = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    for name, n, seed0 in (("train", per_tier, 0), ("heldout", held, 100000)):
        out = ROOT / "maps_gen" / name
        items = generate_set(out, per_tier=n, seed0=seed0)
        (out / "index.json").write_text(json.dumps(items, indent=0))
        print(name, len(items))
