"""Weights & Biases metrics logging for `main_pretrain_tt.py`'s training loop.

Loads `WANDB_API_KEY` from `tenstorrent/.env` (gitignored, already present
on this host) via `python-dotenv`, rather than requiring the key be
pre-exported in the shell -- and never prints/logs the key's value anywhere
(only whether one was found). Everything in this module is only imported/
called when `main()`'s `--wandb` flag is passed (see `main_pretrain_tt.py`):
the default fast pytest smoke-test path never imports `wandb` or reads
`.env`, so it needs no network access or credentials.

User explicitly asked for wandb specifically, not any other metrics
tracker (project memory).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# tenstorrent/src/smri_mae_tt/wandb_tt.py -> parents[2] == tenstorrent/
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_wandb_api_key() -> Optional[str]:
    """Read `WANDB_API_KEY` out of `tenstorrent/.env` without printing it.
    Falls back to an already-exported `WANDB_API_KEY` env var if the `.env`
    file doesn't define one (or doesn't exist). Returns `None` if neither
    source has it -- callers decide what to do with that (see `init_wandb`,
    which fails fast rather than silently running anonymously).
    """
    from dotenv import dotenv_values

    values = dotenv_values(str(_ENV_PATH)) if _ENV_PATH.is_file() else {}
    return values.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY")


def init_wandb(
    *,
    project: str,
    run_name: Optional[str],
    config: dict,
    notes: Optional[str] = None,
    entity: Optional[str] = None,
) -> Any:
    """Authenticate from `tenstorrent/.env` and start a wandb run. Raises
    `RuntimeError` (fail-fast, no silent anonymous/offline fallback) if no
    key is found anywhere -- `--wandb` was explicitly requested, so a
    missing key is a configuration error to surface, not something to work
    around.
    """
    import wandb

    api_key = _load_wandb_api_key()
    if not api_key:
        raise RuntimeError(
            f"WANDB_API_KEY not found in {_ENV_PATH} or the environment, but --wandb was passed. "
            "Not silently falling back to anonymous/offline mode -- set the key in tenstorrent/.env "
            "or pass --no-wandb."
        )
    os.environ.setdefault("WANDB_API_KEY", api_key)
    return wandb.init(project=project, name=run_name, notes=notes, entity=entity, config=config)


def log_train_step(
    run: Any,
    step: int,
    *,
    loss: float,
    lr: float,
    grad_norm: Optional[float],
    replaced_samples: int,
    epoch_equiv: Optional[float] = None,
) -> None:
    if run is None:
        return
    payload: dict[str, float] = {"train/loss": loss, "train/lr": lr, "train/replaced_samples": replaced_samples}
    if grad_norm is not None:
        payload["train/grad_norm"] = grad_norm
    if epoch_equiv is not None:
        payload["train/epoch_equiv"] = epoch_equiv
    run.log(payload, step=step)


def log_eval(run: Any, step: int, *, val_loss: float, num_batches: int) -> None:
    if run is None:
        return
    run.log({"eval/loss": val_loss, "eval/num_batches": num_batches}, step=step)


def finish_wandb(run: Any) -> None:
    if run is not None:
        run.finish()
