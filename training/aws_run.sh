#!/usr/bin/env bash
# One-command training run for a fresh Ubuntu 22.04/24.04 AWS instance (e.g. g5.4xlarge).
#
#   git clone <your repo> && cd <repo> && bash training/aws_run.sh
#
# The game engine runs on the CPU cores (one level per worker); the GPU only does the network
# updates. Everything is resumable: re-run with START_ITER=<n> to continue after an interruption.
# Run it inside tmux so it survives an SSH disconnect:  tmux new -s doom 'bash training/aws_run.sh'
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update -y
sudo apt-get install -y --no-install-recommends git git-lfs curl libgl1 libglib2.0-0 tmux
git lfs install
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv sync --group train

WORKERS="${WORKERS:-$(( $(nproc) > 3 ? $(nproc) - 2 : 1 ))}"
ITERS="${ITERS:-8}"
EPISODES="${EPISODES:-600}"
TRAIN_STEPS="${TRAIN_STEPS:-4000}"
START_ITER="${START_ITER:-0}"

uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# How good are the demonstrations? (the network cannot beat this in the imitation stage)
uv run python -m training.play --policy teacher --seeds 1000000:1000048 --workers "$WORKERS"
uv run python -m training.dagger --workers "$WORKERS" --iters "$ITERS" --episodes "$EPISODES" \
    --train-steps "$TRAIN_STEPS" --start-iter "$START_ITER"
PYTHONUTF8=1 uv run python validate.py --device cpu
echo "Done. Weights: submission/weights/policy.pt   Log: runs/dagger.log"
echo "Commit them with:  git add submission/weights/policy.pt && git commit -m 'Add trained weights' && git push"
