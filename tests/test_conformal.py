import numpy as np

from cascade.conformal import AdaptiveConformal, MondrianConformalRegressor


def test_marginal_coverage_holds():
    rng = np.random.default_rng(0)
    n = 2000
    yhat = rng.normal(0, 1, n)
    y = yhat + rng.normal(0, 0.3, n)  # residual ~ N(0, 0.3)
    groups = ["g"] * n
    n_cal = 1000
    cr = MondrianConformalRegressor(alpha=0.1).fit(y[:n_cal], yhat[:n_cal], groups[:n_cal])
    covered = 0
    for i in range(n_cal, n):
        lo, hi, _ = cr.predict_interval(yhat[i], "g")
        covered += lo <= y[i] <= hi
    cov = covered / (n - n_cal)
    assert cov >= 0.88  # ~ 1 - alpha, with sampling slack


def test_group_conditional_coverage():
    rng = np.random.default_rng(1)
    n = 3000
    g = rng.integers(0, 3, n)
    yhat = rng.normal(0, 1, n)
    # group-specific noise scale
    scales = np.array([0.1, 0.5, 1.2])
    y = yhat + rng.normal(0, scales[g])
    groups = [f"g{x}" for x in g]
    n_cal = 1500
    cr = MondrianConformalRegressor(alpha=0.1).fit(y[:n_cal], yhat[:n_cal], groups[:n_cal])
    for gi in range(3):
        idx = [i for i in range(n_cal, n) if g[i] == gi]
        cov = np.mean([cr.predict_interval(yhat[i], f"g{gi}")[0] <= y[i] <= cr.predict_interval(yhat[i], f"g{gi}")[1] for i in idx])
        assert cov >= 0.86, f"group {gi} coverage {cov}"


def test_unseen_group_falls_back_to_global():
    cr = MondrianConformalRegressor(alpha=0.1).fit(
        np.zeros(50), np.zeros(50), ["seen"] * 50
    )
    _, _, used_global = cr.predict_interval(0.0, "never-seen")
    assert used_global is True
    assert cr.has_support("never-seen") is False
    assert cr.has_support("seen") is True


def test_quantile_rank_is_conformal():
    cr = MondrianConformalRegressor(alpha=0.1)
    scores = np.arange(1, 10)  # n=9
    # ceil((9+1)*0.9)=9 -> 9th smallest (index 8) = 9
    assert cr._quantile(scores, 0.1) == 9
    # n too small -> inf
    assert np.isinf(cr._quantile(np.arange(1, 5), 0.1))


def test_adaptive_conformal_directional_updates():
    # Over-covering (always covered) → ACI raises alpha (asks for narrower sets).
    over = AdaptiveConformal(alpha=0.1, gamma=0.02)
    for _ in range(300):
        over.update(covered=True)
    assert over.effective_alpha() > 0.1

    # Under-covering (always missed) → ACI lowers alpha (asks for wider sets).
    under = AdaptiveConformal(alpha=0.1, gamma=0.02)
    for _ in range(300):
        under.update(covered=False)
    assert under.effective_alpha() < 0.1
    assert 0.0 <= under.effective_alpha() <= 1.0


def test_adaptive_conformal_self_corrects_to_target():
    # Coverage outcome depends on the current alpha_t: P(cover) = 1 - alpha_t.
    # ACI should drive realized coverage toward the 1 - alpha target.
    rng = np.random.default_rng(0)
    aci = AdaptiveConformal(alpha=0.1, gamma=0.03)
    for _ in range(5000):
        covered = rng.uniform() < (1 - aci.effective_alpha())
        aci.update(covered)
    assert abs(aci.realized_coverage() - 0.9) < 0.03
