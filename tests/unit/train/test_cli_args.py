"""Tests for CLI arguments."""

import pytest

from scripts.train import parse_args
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.metrics import ce_loss, kl_div_loss
from speculators.models.peagle.core import PEagleDraftModel


def _parse(monkeypatch, extra: list[str]):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--verifier-name-or-path", "dummy"] + extra
    )
    return parse_args()


# ---------------------------------------------------------------------------
# Ensure CLI args flow correctly through vars(args) into get_trainer_kwargs
# ---------------------------------------------------------------------------


def test_dflash_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is kl_div_loss
    assert val_kw["loss_fn"] is kl_div_loss


def test_dflash_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is ce_loss
    assert val_kw["loss_fn"] is ce_loss


def test_dspark_default_loss_weights(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw == val_kw
    assert train_kw["ce_loss_alpha"] == pytest.approx(0.1)
    assert train_kw["l1_loss_alpha"] == pytest.approx(0.9)
    assert train_kw["confidence_head_alpha"] == pytest.approx(1.0)
    assert train_kw["loss_decay_gamma"] == pytest.approx(4.0)


def test_dspark_explicit_loss_weights(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--ce-loss-alpha",
            "0.2",
            "--l1-loss-alpha",
            "0.7",
            "--confidence-head-alpha",
            "0.5",
            "--loss-decay-gamma",
            "6.0",
        ],
    )
    train_kw, _val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["ce_loss_alpha"] == pytest.approx(0.2)
    assert train_kw["l1_loss_alpha"] == pytest.approx(0.7)
    assert train_kw["confidence_head_alpha"] == pytest.approx(0.5)
    assert train_kw["loss_decay_gamma"] == pytest.approx(6.0)


def test_eagle3_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is kl_div_loss
    assert val_kw["loss_fn"] is kl_div_loss


def test_eagle3_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is ce_loss
    assert val_kw["loss_fn"] is ce_loss


def test_peagle_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is kl_div_loss
    assert val_kw["loss_fn"] is kl_div_loss


def test_peagle_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_fn"] is ce_loss
    assert val_kw["loss_fn"] is ce_loss
