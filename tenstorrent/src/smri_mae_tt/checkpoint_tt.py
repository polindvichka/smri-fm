"""Checkpoint save/load for `main_pretrain_tt.py`'s training loop.

Uses `params.py`'s `tt_to_state_dict`/`state_dict_to_tt` (already
implemented and tested -- see `test_params.py`'s round-trip test) for the
TT<->torch `state_dict` conversion, plus plain `torch.save`/`torch.load`
for on-disk persistence -- the same pattern the PyTorch reference recipe
uses (`main_pretrain.py`'s own `torch.save(..., ckpt_path)`), just
retargeted at this port's `Encoder`/`Decoder` pair instead of a single
`MaskedAutoencoderViT` module.

## What is (and isn't) in a checkpoint here

Only encoder/decoder *weights* (`params.tt_to_state_dict`'s output) plus a
small metadata envelope (`step`, and anything the caller passes as
`extra`). Optimizer state (`MultiGroupAdamW`'s per-group
`AdamWFullPrecision` `exp_avg`/`exp_avg_sq`/master-weight buffers) is
**not** saved -- `ttml.optimizers.AdamWFullPrecision` exposes no
Python-level state_dict/serialization hook this module could use (grepped
the built wheel's bindings; none exists), so resuming from a checkpoint
here means resuming *weights only*, with a freshly-initialized optimizer
state (equivalent to a warm-start, not a bit-exact mid-run resume). This is
documented here rather than silently pretended away.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from smri_mae_tt.params import state_dict_to_tt, tt_to_state_dict

if TYPE_CHECKING:
    from smri_mae_tt.modules_tt import Decoder, Encoder


def checkpoint_path(checkpoint_dir: str, step: int) -> str:
    return os.path.join(checkpoint_dir, f"step_{step:08d}.pt")


def save_checkpoint(
    encoder: "Encoder",
    decoder: "Decoder",
    step: int,
    checkpoint_dir: str,
    *,
    extra: Optional[dict] = None,
) -> str:
    """Write `encoder`/`decoder` state (via `params.tt_to_state_dict`) plus
    a small metadata envelope to `checkpoint_dir/step_<N>.pt`. Creates
    `checkpoint_dir` if missing (`os.makedirs(..., exist_ok=True)` is the
    one directory-creation convenience allowed here, mirroring
    `main_pretrain.py`'s own `output_dir.mkdir(parents=True,
    exist_ok=True)`) -- everything else (a bad `encoder`/`decoder`, a
    non-writable path) raises through `torch.save`/`tt_to_state_dict`
    unmodified, not caught here.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = tt_to_state_dict(encoder, decoder)
    payload = {"step": step, "state_dict": state_dict}
    if extra:
        overlap = set(extra) & {"step", "state_dict"}
        if overlap:
            raise ValueError(f"extra checkpoint payload keys collide with reserved keys: {sorted(overlap)}")
        payload.update(extra)
    path = checkpoint_path(checkpoint_dir, step)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, encoder: "Encoder", decoder: "Decoder") -> dict:
    """Load a checkpoint written by `save_checkpoint` into an
    already-constructed `encoder`/`decoder` pair (must already match the
    checkpoint's `ShapeContract`/depths -- `state_dict_to_tt` fails fast
    with a clear `KeyError`/`ValueError` on any mismatch, exactly as it
    does for any other malformed `state_dict`; nothing here downgrades or
    catches that). Returns the full checkpoint payload dict (so callers can
    read back `step`/`extra` fields) after applying the weights in place.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in payload:
        raise KeyError(f"checkpoint {path!r} has no 'state_dict' key -- not a save_checkpoint() output")
    state_dict_to_tt(payload["state_dict"], encoder, decoder)
    return payload
