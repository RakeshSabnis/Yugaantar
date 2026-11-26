"""Copy a checkpoint into submission/weights/policy.pt (model weights only).

    uv run python training/publish.py runs/dagger1/best.pt
"""
import sys
from pathlib import Path

import torch

src = sys.argv[1]
s = torch.load(src, map_location="cpu", weights_only=False)
dst = Path(__file__).resolve().parents[1] / "submission" / "weights" / "policy.pt"
torch.save({"model": s["model"]}, dst)
h = s.get("heldout") or {}
print(f"published {src} -> {dst} (held-out weighted {h.get('weighted', float('nan')):.1f})")
