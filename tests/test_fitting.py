"""Fitting: noise-free recovery, recorded failures, at-bound flags, standard
errors and the fits table schema."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from drydown_q.fitting import fit_all, fit_event, parameter_setup
from drydown_q.models import exponential, q_stage2
from drydown_q.schemas import assert_schema

SENSOR = {"theta_fc_m3m3": 0.32, "min_sm_m3m3": 0.04, "theta_w_m3m3": 0.05, "dz_m": 0.05}
T = np.arange(12, dtype=float)
PET = np.full(12, 5.0)


def test_parameter_setup_follows_morgan(smoke_cfg):
    theta = q_stage2(T, 0.30, 0.05, 0.03, 1.2, 0.28)
    fixed, free = parameter_setup("q_stage2", theta, PET, SENSOR, smoke_cfg.fitting)
    assert fixed == {"theta_0": pytest.approx(0.30), "theta_w": 0.05}
    assert list(free) == ["k", "q", "theta_star"]
    assert free["k"] == pytest.approx((0, 0.1, 0.05))          # PET_max / (1000 dz), start half
    assert free["q"] == (0, float("inf"), 1.001)
    assert free["theta_star"] == (0.04, 0.32, 0.32)             # [min_sm, theta_fc], start upper
    # theta_0 is capped at theta_fc; missing dPET uses pet_default_mm_day.
    fixed, free = parameter_setup("q_stage2", theta + 0.1, np.full(12, np.nan), SENSOR, smoke_cfg.fitting)
    assert fixed["theta_0"] == 0.32 and free["k"][1] == pytest.approx(0.1)


def test_noise_free_q_fit_reproduces_the_curve(smoke_cfg):
    theta = q_stage2(T, 0.30, 0.05, 0.03, 1.4, 0.28)
    row = fit_event(T, theta, PET, SENSOR, "q_stage2", smoke_cfg.fitting)
    assert row["fit_status"] == "ok"
    assert row["r2"] > 0.999999 and row["rmse_m3m3"] < 1e-5
    assert row["q"] == pytest.approx(1.4, rel=1e-3)
    assert row["et_max_mm_day"] == pytest.approx(1000 * 0.05 * row["k_per_day"])
    assert {"at_bound_k_per_day", "at_bound_q", "at_bound_theta_star_m3m3", "q_se"} <= set(row)


def test_noise_free_exponential_fit(smoke_cfg):
    theta = exponential(T, 0.30, 0.045, 4.0)
    row = fit_event(T, theta, PET, SENSOR, "exponential", smoke_cfg.fitting)
    assert row["fit_status"] == "ok"
    assert row["tau_days"] == pytest.approx(4.0, rel=1e-4)
    assert row["theta_w_m3m3"] == pytest.approx(0.045, rel=1e-4)
    assert row["k_per_day"] == pytest.approx((0.32 - row["theta_w_m3m3"]) / row["tau_days"])


def test_degenerate_bounds_are_recorded_not_raised(smoke_cfg):
    # exponential theta_w bounds [min_sm, event_min] collapse when the event
    # reaches min_sm: Morgan's fit failed here, and ours records the error.
    theta = exponential(T, 0.30, 0.04, 2.0)
    theta[-1] = SENSOR["min_sm_m3m3"]
    row = fit_event(T, theta, PET, SENSOR, "exponential", smoke_cfg.fitting)
    assert row["fit_status"] == "error" and "ValueError" in row["fit_message"]
    assert np.isnan(row["r2"])


def test_at_bound_flag_from_active_set(smoke_cfg):
    # True theta_w = 0 lies below the lower bound min_sm = 0.04: the optimum
    # is outside the box, so theta_w is pinned and the active set says so.
    t = T[:6]
    theta = exponential(t, 0.30, 0.0, 3.0)
    row = fit_event(t, theta, PET[:6], SENSOR, "exponential", smoke_cfg.fitting)
    assert row["fit_status"] == "ok", row["fit_message"]
    assert row["at_bound_theta_w_m3m3"] and row["theta_w_m3m3"] == pytest.approx(0.04)
    assert not row["at_bound_tau_days"]


def test_bound_approached_but_not_active_is_not_flagged(smoke_cfg):
    # A linear decline (q = 0) is fitted with q slightly above 0; the solver
    # stops on its gradient tolerance before the bound becomes active.
    row = fit_event(T, 0.30 - 0.01 * T, PET, SENSOR, "q_stage2", smoke_cfg.fitting)
    assert row["fit_status"] == "ok" and row["q"] < 0.01 and not row["at_bound_q"]


def test_standard_errors_are_inf_without_degrees_of_freedom(smoke_cfg):
    theta = q_stage2(T[:3], 0.30, 0.05, 0.03, 1.4, 0.28)
    row = fit_event(T[:3], theta, PET[:3], SENSOR, "q_stage2", smoke_cfg.fitting)
    assert np.isinf(row["q_se"])


def test_fixed_q_parameter_uses_its_start(smoke_cfg):
    cfg = dataclasses.replace(smoke_cfg.fitting, free_params_q=["k", "q"])
    theta = q_stage2(T, 0.30, 0.05, 0.03, 1.4, 0.32)
    row = fit_event(T, theta, PET, SENSOR, "q_stage2", cfg)
    assert row["theta_star_m3m3"] == 0.32 and "at_bound_theta_star_m3m3" not in row
    assert row["q"] == pytest.approx(1.4, rel=1e-4)


def test_fit_all_rows_and_schema(smoke_cfg):
    events = pd.DataFrame({"event_id": ["a/1", "a/2", "b/1"], "sensor_id": ["a", "a", "b"]})
    days = []
    for event_id, q in [("a/1", 0.8), ("a/2", 1.2), ("b/1", 2.0)]:
        theta = q_stage2(T, 0.30, 0.05, 0.03, q, 0.28)
        days.append(pd.DataFrame({"event_id": event_id, "t_days": T.astype(int),
                                  "theta_m3m3": theta, "pet_mm": PET}))
    days = pd.concat(days)
    days.loc[days.index[3], "theta_m3m3"] = np.nan          # a missing day is skipped
    constants = pd.DataFrame([{"sensor_id": s, **SENSOR} for s in ["a", "b"]])
    fits = fit_all(events, days, constants, smoke_cfg.fitting, n_jobs=1)
    assert len(fits) == 3 * len(smoke_cfg.fitting.models)
    assert (fits["fit_status"] == "ok").all()
    assert_schema(fits, "fits")


def test_piecewise_model_recovers_stage1_and_stage2(smoke_cfg):
    # theta_0 above theta_star: a linear stage 1 at rate k, then stage 2. Here k and
    # theta_star are identifiable (unlike in q_stage2), because the kink fixes theta_star.
    from drydown_q.models import q_piecewise
    cfg = dataclasses.replace(smoke_cfg.fitting, models=["q_piecewise"], theta_star_start="midpoint")
    t = np.arange(16, dtype=float)
    theta = q_piecewise(t, 0.31, 0.05, 0.03, 1.6, 0.22)
    row = fit_event(t, theta, np.full(t.size, 5.0), SENSOR, "q_piecewise", cfg)
    assert row["fit_status"] == "ok" and row["r2"] > 0.9999
    assert row["k_per_day"] == pytest.approx(0.03, rel=1e-3)
    assert row["theta_star_m3m3"] == pytest.approx(0.22, rel=1e-3)
    assert row["q"] == pytest.approx(1.6, rel=1e-2)
    # Started at the upper bound (theta_star = theta_fc > theta_0) it stops in a local minimum.
    stuck = fit_event(t, theta, np.full(t.size, 5.0), SENSOR, "q_piecewise",
                      dataclasses.replace(cfg, theta_star_start="upper"))
    assert stuck["theta_star_m3m3"] > 0.3


def test_analysis_model_must_be_fitted():
    from drydown_q.config import ConfigError, load_config
    from conftest import SMOKE_PATHS
    with pytest.raises(ConfigError, match="analysis_model"):
        load_config("smoke", SMOKE_PATHS, overrides={"covariates": {"analysis_model": "q_piecewise"}})
    cfg = load_config("smoke", SMOKE_PATHS, overrides={"covariates": {"analysis_model": "q_piecewise"},
                                                      "fitting": {"models": ["exponential", "q_stage2", "q_piecewise"]}})
    assert cfg.covariates.analysis_model == "q_piecewise"
