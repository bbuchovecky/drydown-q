"""Each analytical solution matches numerical integration of its ODE
(relative error < 1e-6) over a grid of parameters."""

import itertools

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from drydown_q import models

THETA_W, THETA_STAR = 0.05, 0.30
T = np.linspace(0, 20, 81)


def integrate(rate, theta_0, t_end, **params):
    sol = solve_ivp(lambda t, y: rate(y, **params), (0, t_end), [theta_0], t_eval=T[T <= t_end],
                    method="DOP853", rtol=1e-12, atol=1e-14)
    return sol.t, sol.y[0]


def assert_close(analytic, numeric):
    rel = np.abs(analytic - numeric) / np.abs(numeric)
    assert rel.max() < 1e-6, rel.max()


@pytest.mark.parametrize("theta_0, tau", list(itertools.product([0.15, 0.30, 0.40], [0.5, 3.0, 20.0])))
def test_exponential_matches_ode(theta_0, tau):
    t, numeric = integrate(models.exponential_rate, theta_0, T[-1], theta_w=THETA_W, tau_days=tau)
    assert_close(models.exponential(t, theta_0, THETA_W, tau), numeric)


def _t_end(theta_0, k, q):
    """Stay before theta reaches theta_w (finite time when q < 1)."""
    if q >= 1:
        return T[-1]
    s = THETA_STAR - THETA_W
    t_zero = (theta_0 - THETA_W) ** (1 - q) * s ** q / (k * (1 - q))
    return min(T[-1], 0.9 * t_zero)


Q_GRID = list(itertools.product([0.12, 0.25, 0.30], [0.005, 0.02, 0.08], [0.3, 0.7, 1.0, 1.001, 1.5, 3.0]))


@pytest.mark.parametrize("theta_0, k, q", Q_GRID)
def test_q_stage2_matches_ode(theta_0, k, q):
    params = dict(theta_w=THETA_W, k_per_day=k, q=q, theta_star=THETA_STAR)
    t, numeric = integrate(models.q_stage2_rate, theta_0, _t_end(theta_0, k, q), **params)
    assert_close(models.q_stage2(t, theta_0, THETA_W, k, q, THETA_STAR), numeric)


@pytest.mark.parametrize("theta_0, k, q", list(itertools.product([0.25, 0.35, 0.42], [0.01, 0.04], [0.5, 1.0, 2.0])))
def test_q_piecewise_matches_ode(theta_0, k, q):
    params = dict(theta_w=THETA_W, k_per_day=k, q=q, theta_star=THETA_STAR)
    t_star = max(theta_0 - THETA_STAR, 0) / k
    t_end = t_star + _t_end(min(theta_0, THETA_STAR), k, q)
    t_end = min(t_end, T[-1])
    # Integrate the two stages separately so the kink at theta_star is exact.
    t1 = T[T <= min(t_star, t_end)]
    stage1 = theta_0 - k * t1
    sol = solve_ivp(lambda t, y: models.q_stage2_rate(y, **params), (t_star, t_end),
                    [min(theta_0, THETA_STAR)], t_eval=T[(T > t_star) & (T <= t_end)],
                    method="DOP853", rtol=1e-12, atol=1e-14)
    t = np.concatenate([t1, sol.t])
    numeric = np.concatenate([stage1, sol.y[0]])
    assert_close(models.q_piecewise(t, theta_0, THETA_W, k, q, THETA_STAR), numeric)


def test_piecewise_rate_is_minus_k_above_theta_star():
    assert models.q_piecewise_rate(0.35, THETA_W, 0.04, 2.0, THETA_STAR) == -0.04
    assert models.q_piecewise_rate(0.175, THETA_W, 0.04, 1.0, THETA_STAR) == pytest.approx(-0.02)


def test_q_stage2_is_nan_after_reaching_theta_w_for_q_below_one():
    # Morgan's behaviour, kept: the bracket goes negative after finite time.
    theta = models.q_stage2(np.array([0, 1000]), 0.2, THETA_W, 0.05, 0.6, THETA_STAR)
    assert np.isfinite(theta[0]) and np.isnan(theta[1])
    # When 1/(1-q) is an integer the power of a negative bracket is real, so the
    # curve rises again past theta_w instead of becoming NaN (also Morgan's).
    theta = models.q_stage2(np.array([1000]), 0.2, THETA_W, 0.05, 0.5, THETA_STAR)
    assert theta[0] > 0.2
