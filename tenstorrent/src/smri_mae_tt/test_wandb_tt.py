"""Tests for `wandb_tt.py`'s gating: the fast pytest path must never attempt
a real wandb network call.

`log_train_step`/`log_eval`/`finish_wandb` all no-op on `run=None`, which is
what `main_pretrain_tt.train`'s `wandb_run` parameter defaults to -- so
these are checked directly here, without importing the real `wandb` package
network path at all. `init_wandb` (the only function that talks to wandb.ai)
is exercised separately, mocked, to confirm it reads the key from
`tenstorrent/.env` without ever printing/logging it -- not a live network
test (no `--wandb`-gated live run is part of this fast suite; see
`test_main_pretrain_tt.py`'s module docstring / task report for why a real
run was or wasn't exercised).
"""

from __future__ import annotations

from unittest import mock

import pytest

from smri_mae_tt import wandb_tt


def test_log_functions_noop_on_none_run(capsys):
    """No exception, no output, and critically no attempt to import/talk to
    the real `wandb` package when `run=None` (the default `train()` uses
    unless `--wandb` is passed)."""
    wandb_tt.log_train_step(None, 0, loss=1.0, lr=0.1, grad_norm=0.5, replaced_samples=0)
    wandb_tt.log_eval(None, 0, val_loss=1.0, num_batches=3)
    wandb_tt.finish_wandb(None)
    # No stdout/stderr chatter from these no-ops.
    captured = capsys.readouterr()
    assert captured.out == ""


def test_log_train_step_forwards_to_run_log():
    run = mock.Mock()
    wandb_tt.log_train_step(run, 5, loss=1.23, lr=0.01, grad_norm=2.0, replaced_samples=1, epoch_equiv=0.5)
    run.log.assert_called_once()
    (payload,), kwargs = run.log.call_args
    assert payload["train/loss"] == 1.23
    assert payload["train/lr"] == 0.01
    assert payload["train/grad_norm"] == 2.0
    assert payload["train/replaced_samples"] == 1
    assert payload["train/epoch_equiv"] == 0.5
    assert kwargs["step"] == 5


def test_log_train_step_omits_grad_norm_and_epoch_equiv_when_none():
    run = mock.Mock()
    wandb_tt.log_train_step(run, 0, loss=1.0, lr=0.1, grad_norm=None, replaced_samples=0)
    (payload,), _ = run.log.call_args
    assert "train/grad_norm" not in payload
    assert "train/epoch_equiv" not in payload


def test_log_eval_forwards_to_run_log():
    run = mock.Mock()
    wandb_tt.log_eval(run, 10, val_loss=0.5, num_batches=4)
    run.log.assert_called_once_with({"eval/loss": 0.5, "eval/num_batches": 4}, step=10)


def test_finish_wandb_calls_run_finish():
    run = mock.Mock()
    wandb_tt.finish_wandb(run)
    run.finish.assert_called_once()


def test_init_wandb_raises_without_api_key(monkeypatch, tmp_path):
    """No key in .env or the environment -> fail fast, no silent
    anonymous/offline wandb.init(). Points `_ENV_PATH` at an empty temp dir
    so this test is independent of whether the real tenstorrent/.env exists
    on this host."""
    monkeypatch.setattr(wandb_tt, "_ENV_PATH", tmp_path / "nonexistent.env")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        wandb_tt.init_wandb(project="p", run_name="r", config={})


def test_init_wandb_reads_key_from_env_file_without_printing_it(monkeypatch, tmp_path, capsys):
    env_file = tmp_path / "test.env"
    secret = "wandb_v1_totally_fake_key_for_this_test_only"
    env_file.write_text(f"WANDB_API_KEY={secret}\n")
    monkeypatch.setattr(wandb_tt, "_ENV_PATH", env_file)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    fake_wandb = mock.Mock()
    fake_run = mock.Mock()
    fake_wandb.init.return_value = fake_run

    with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
        run = wandb_tt.init_wandb(project="p", run_name="r", config={"a": 1})

    assert run is fake_run
    fake_wandb.init.assert_called_once_with(project="p", name="r", notes=None, entity=None, config={"a": 1})

    import os

    assert os.environ.get("WANDB_API_KEY") == secret
    monkeypatch.delenv("WANDB_API_KEY", raising=False)  # avoid leaking into later tests

    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
