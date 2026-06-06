import numpy as np

from cascade.baselines import beats_baseline_gate, mean_baseline, ridge_baseline, mse


def test_mean_model_does_not_beat_mean():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 200)
    mean_pred = mean_baseline(y, len(y))
    ridge_pred = mean_pred  # irrelevant here
    gate = beats_baseline_gate(y, model_pred=mean_pred, mean_pred=mean_pred, ridge_pred=ridge_pred)
    assert gate.beats_mean is True  # ties allowed (<=)
    # a strictly worse model does not promote
    worse = mean_pred + 5
    gate2 = beats_baseline_gate(y, worse, mean_pred, ridge_pred, rel_margin=0.0)
    assert gate2.promote is False


def test_perfect_model_promotes():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 2, 300)
    mean_pred = mean_baseline(y, len(y))
    gate = beats_baseline_gate(y, model_pred=y, mean_pred=mean_pred, ridge_pred=mean_pred)
    assert gate.promote is True
    assert gate.model_mse == 0.0


def test_ridge_beats_mean_on_linear_data():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (400, 5))
    w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = X @ w + rng.normal(0, 0.1, 400)
    Xtr, ytr, Xte, yte = X[:300], y[:300], X[300:], y[300:]
    rp = ridge_baseline(Xtr, ytr, Xte)
    mp = mean_baseline(ytr, len(yte))
    assert mse(yte, rp) < mse(yte, mp)


def test_low_variance_warning():
    y = np.full(100, 3.0) + np.random.default_rng(0).normal(0, 1e-3, 100)
    gate = beats_baseline_gate(y, y, mean_baseline(y, 100), y)
    assert gate.low_variance_warning is True


def test_rel_margin_requires_real_improvement():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 200)
    model = y + rng.normal(0, 0.5, 200)  # decent but not perfect
    mean_pred = mean_baseline(y, 200)
    # require 99% improvement -> should fail
    gate = beats_baseline_gate(y, model, mean_pred, mean_pred, rel_margin=0.99)
    assert gate.promote is False
