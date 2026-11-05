"""Synthetic parameter recovery (marker ``recovery``; run with ``pytest -m recovery -s``).

Drydowns are generated from the q model with known parameters, plus Gaussian
noise of sigma = 0.005-0.02 m3 m-3, over 5-20 days, and fitted with the
baseline ``fitting`` config unchanged. The test reports bias and spread of the
recovered q and the share of fits with any parameter at a bound, and writes
``outputs/recovery/recovery_q.csv``.

For now the events are given (step 3 only). Once events.py exists (build phase
5) the series also go through step 2, event detection, unchanged.

Generated truth: the physically continued solution (theta stays at theta_w
after reaching it when q < 1), not the NaN/rebound form of the fitted model.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from drydown_q.fitting import fit_event
from conftest import REPO

THETA_W, THETA_STAR, THETA_0, K = 0.05, 0.28, 0.30, 0.04
SENSOR = {"theta_fc_m3m3": 0.32, "min_sm_m3m3": 0.04, "theta_w_m3m3": THETA_W, "dz_m": 0.05}
PET_MM = 5.0
Q_TRUE = [0.5, 0.8, 1.0, 1.3, 1.8, 2.5]
SIGMAS = [0.005, 0.01, 0.02]
LENGTHS = [5, 10, 15, 20]
REPLICATES = 40


def truth(t, q):
    s = THETA_STAR - THETA_W
    if q == 1:
        return THETA_W + (THETA_0 - THETA_W) * np.exp(-K * t / s)
    bracket = np.maximum((THETA_0 - THETA_W) ** (1 - q) - K * (1 - q) / s ** q * t, 0)
    return bracket ** (1 / (1 - q)) + THETA_W


@pytest.mark.recovery
def test_q_recovery(smoke_cfg):
    rng = np.random.default_rng(42)
    rows = []
    for q, sigma, length in itertools.product(Q_TRUE, SIGMAS, LENGTHS):
        t = np.arange(length + 1, dtype=float)
        for _ in range(REPLICATES):
            theta = truth(t, q) + rng.normal(0, sigma, t.size)
            fit = fit_event(t, theta, np.full(t.size, PET_MM), SENSOR, "q_stage2", smoke_cfg.fitting)
            at_bound = [v for k, v in fit.items() if k.startswith("at_bound_")]
            rows.append({"q_true": q, "sigma": sigma, "length_days": length,
                         "q_hat": fit.get("q", np.nan), "status": fit["fit_status"],
                         "r2": fit["r2"], "any_at_bound": bool(any(at_bound)),
                         "q_at_bound": bool(fit.get("at_bound_q", False))})
    fits = pd.DataFrame(rows)
    fits["error"] = fits["q_hat"] - fits["q_true"]
    summary = (fits.groupby(["q_true", "sigma", "length_days"])
               .agg(n=("error", "size"), share_ok=("status", lambda s: (s == "ok").mean()),
                    bias_median=("error", "median"),
                    spread_iqr=("error", lambda e: e.quantile(0.75) - e.quantile(0.25)),
                    share_any_at_bound=("any_at_bound", "mean"),
                    share_q_at_bound=("q_at_bound", "mean"),
                    share_r2_ge_min=("r2", lambda r: (r >= smoke_cfg.filters.r2_min).mean()))
               .reset_index())
    out = REPO / "outputs" / "recovery"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "recovery_q.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n", summary.round(3).to_string(index=False))

    # Sanity only: every fit is recorded (failures included), and the easiest
    # cases recover q wherever fits run. Known failure, reported not asserted:
    # for q < 1 the fitted model is NaN past theta_w, so the solver fails on
    # long low-q events (Morgan's behaviour; see models.q_stage2).
    assert len(fits) == len(Q_TRUE) * len(SIGMAS) * len(LENGTHS) * REPLICATES
    easy = summary[(summary.sigma == min(SIGMAS)) & (summary.length_days == max(LENGTHS))
                   & (summary.share_ok == 1)]
    assert len(easy) >= len(Q_TRUE) - 1
    assert (easy["bias_median"].abs() < 0.2).all()
