#!/usr/bin/env bash
# One-time (and re-runnable/incremental) setup of the TT dev environment,
# directly on this host -- no Docker. This machine already has the
# Tenstorrent driver, firmware, and hugepages configured (tt-installer),
# and tt-metal's build officially supports Ubuntu 24.04 + Python 3.12, which
# is exactly what this host is.
#
# What this does:
#   1. Clones tenstorrent/tt-metal into tenstorrent/tt-metal (gitignored;
#      pinned commit recorded in tenstorrent/tt-metal.commit) if not present,
#      pins v0.75.0, then applies tenstorrent/patches/vendored-tt-metal-changes.patch
#      (TENSTORRENT_PORT.md section 7's vendored SDPA changes).
#   2. sudo ./install_dependencies.sh -- apt packages, LLVM 20, SFPI, OpenMPI
#      ULFM, CMake. Modifies host system state (this is the one host-invasive
#      step; everything else is scoped to the tt-metal checkout).
#   3. ./build_metal.sh --build-tt-train -- builds tt-metal + tt-train (ttml).
#      This is the slow step (C++ build).
#   4. ./create_venv.sh -- creates tenstorrent/tt-metal/python_env with ttnn +
#      ttml wired in via .pth files.
#   5. uv pip install -e tenstorrent/ (its own project, tenstorrent/pyproject.toml)
#      into that same venv. It depends on the root smri-fm package (path dep),
#      so torch, the dataloader, and smri_mae_tt all end up importable
#      alongside ttnn/ttml.
#
# Safe to re-run: build_metal.sh and create_venv.sh are incremental/idempotent.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TT_METAL_DIR="$ROOT/tt-metal"

echo "[1/5] tt-metal checkout"
if [ ! -d "$TT_METAL_DIR/.git" ]; then
    git clone https://github.com/tenstorrent/tt-metal.git \
        --recurse-submodules --shallow-submodules --depth 1 "$TT_METAL_DIR"
fi
# Pin to v0.75.0 for reproducibility, both on a fresh clone and when re-run
# against an existing checkout (bumped from v0.74.0, 2026-08-05 stability
# pass -- see TENSTORRENT_PORT.md section 1).
git -C "$TT_METAL_DIR" fetch --depth 1 origin tag v0.75.0
git -C "$TT_METAL_DIR" checkout v0.75.0
git -C "$TT_METAL_DIR" submodule sync --recursive
git -C "$TT_METAL_DIR" submodule update --init --recursive --depth 1

# Apply this project's vendored tt-train changes on top of the bare v0.75.0
# tag (TENSTORRENT_PORT.md section 7): AttentionMaskType::None (bidirectional)
# support through the ops-layer wrapper + nanobind binding (M0 fix, required
# for this model), and B17's sfpu_reduce row-sum fusion in the SDPA backward
# kernel (kept from the M6 search, no hang history). Idempotent: skipped if
# already applied (re-running setup.sh against an existing checkout).
PATCH="$ROOT/patches/vendored-tt-metal-changes.patch"
if git -C "$TT_METAL_DIR" apply --check "$PATCH" 2>/dev/null; then
    git -C "$TT_METAL_DIR" apply "$PATCH"
    echo "  applied $PATCH"
elif git -C "$TT_METAL_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "  $PATCH already applied, skipping"
else
    echo "  ERROR: $PATCH does not apply cleanly to v0.75.0 -- tt-metal checkout" \
         "or the patch is out of sync, needs manual reconciliation" >&2
    exit 1
fi

git -C "$TT_METAL_DIR" rev-parse HEAD > "$ROOT/tt-metal.commit"
echo "v0.75.0 + vendored patch (uncommitted working-tree diff, see $PATCH)" >> "$ROOT/tt-metal.commit"
echo "  pinned at $(head -1 "$ROOT/tt-metal.commit")"

echo "[2/5] install_dependencies.sh (sudo, modifies host packages)"
sudo "$TT_METAL_DIR/install_dependencies.sh"

echo "[3/5] build tt-metal + tt-train (this is the slow step)"
(cd "$TT_METAL_DIR" && ./build_metal.sh --build-tt-train)

echo "[4/5] python venv (ttnn + ttml)"
(cd "$TT_METAL_DIR" && ./create_venv.sh)

echo "[5/5] layer this project's dependencies into that venv (uv)"
uv pip install --python "$TT_METAL_DIR/python_env/bin/python" -e "$ROOT"
# `uv pip install` (pip-compat mode) does not honor tenstorrent/pyproject.toml's
# [tool.uv.sources]/[[tool.uv.index]] torch->CPU-wheel pin (that's only
# resolved by `uv sync`/project commands), so it re-pulls the CUDA wheel +
# ~4GB of unused nvidia-* libs -- this host has no NVIDIA GPU (Tenstorrent
# Blackhole only). Force the CPU wheel explicitly and drop the CUDA leftovers.
uv pip install --python "$TT_METAL_DIR/python_env/bin/python" \
    --index-url https://download.pytorch.org/whl/cpu \
    --reinstall-package torch --reinstall-package torchvision \
    "torch==2.8.0" "torchvision==0.23.0"
nvidia_pkgs="$(uv pip list --python "$TT_METAL_DIR/python_env/bin/python" 2>/dev/null | awk '/^nvidia-/{print $1}')"
if [ -n "$nvidia_pkgs" ]; then
    # shellcheck disable=SC2086
    uv pip uninstall --python "$TT_METAL_DIR/python_env/bin/python" $nvidia_pkgs triton
fi

echo "done. Activate with: source $TT_METAL_DIR/python_env/bin/activate"
echo "Then: export TT_METAL_HOME=$TT_METAL_DIR PYTHONPATH=\$TT_METAL_HOME"
