"""Filters: each rule's boundary (inclusive or strict), NaN handling, inactive
rules, and that the cascade counts add up."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from drydown_q.filters import RULES, cascade, flag_filters


def _event(**overrides):
    """One event that passes every default rule."""
    row = dict(event_id="e", station_id="s", r2=0.9, duration_days=10, q=1.0, frac_range=0.5,
               mean_depth_m=0.05, dz_m=0.05, theta_star_m3m3=0.3, min_sm_m3m3=0.05,
               theta_w_m3m3=0.05, lat_deg=40.0, veg_class="HERB", lai_m2m2=1.0)
    row.update(overrides)
    return row


def _flags(cfg, **overrides):
    return flag_filters(pd.DataFrame([_event(**overrides)]), cfg).iloc[0]


def test_reference_event_passes_everything(smoke_cfg):
    flags = _flags(smoke_cfg.filters)
    assert all(flags[f"pass_{r}"] for r in RULES) and flags["in_final_set"]


@pytest.mark.parametrize("column, value, rule, passes", [
    ("r2", 0.7, "r2", True),                 # r2 >= 0.7 inclusive
    ("r2", 0.6999, "r2", False),
    ("duration_days", 6, "duration", True),  # [6, 100] inclusive
    ("duration_days", 5, "duration", False),
    ("duration_days", 100, "duration", True),
    ("duration_days", 101, "duration", False),
    ("q", 0.01, "q_min", False),             # q > 0.01 strict
    ("q", 0.0101, "q_min", True),
    ("frac_range", 0.2, "frac_range", False),  # strict
    ("mean_depth_m", 0.5, "mean_depth", True),
    ("mean_depth_m", 0.51, "mean_depth", False),
    ("dz_m", 0.5, "dz", True),
    ("dz_m", 0.51, "dz", False),
    ("theta_star_m3m3", 0.0549, "theta_star_minus_min_sm", False),
    ("veg_class", "SHR", "class", False),
    ("lai_m2m2", 0.5, "lai", True),          # LAI >= 0.5 (lai_min_inclusive)
    ("lai_m2m2", 0.49, "lai", False),
    ("r2", np.nan, "r2", False),             # NaN fails
    ("lai_m2m2", np.nan, "lai", False),
])
def test_rule_boundaries(smoke_cfg, column, value, rule, passes):
    flags = _flags(smoke_cfg.filters, **{column: value})
    assert bool(flags[f"pass_{rule}"]) is passes
    assert bool(flags["in_final_set"]) is passes


def test_theta_star_minus_w_boundary(smoke_cfg):
    assert _flags(smoke_cfg.filters, theta_star_m3m3=0.30, theta_w_m3m3=0.296)["pass_theta_star_minus_w"] == False  # noqa: E712
    assert _flags(smoke_cfg.filters, theta_star_m3m3=0.30, theta_w_m3m3=0.29)["pass_theta_star_minus_w"] == True  # noqa: E712


def test_strict_lai_option(smoke_cfg):
    cfg = dataclasses.replace(smoke_cfg.filters, lai_min_inclusive=False)
    assert not _flags(cfg, lai_m2m2=0.5)["pass_lai"]


def test_abs_lat_inactive_when_null_and_active_when_set(smoke_cfg):
    assert _flags(smoke_cfg.filters, lat_deg=80.0)["pass_abs_lat"]
    cfg = dataclasses.replace(smoke_cfg.filters, abs_lat_max_deg=60.0)
    assert not _flags(cfg, lat_deg=-80.0)["pass_abs_lat"]


def test_flags_never_drop_rows(smoke_cfg):
    rows = [_event(event_id=str(i), r2=r2) for i, r2 in enumerate([0.1, 0.9, np.nan])]
    out = flag_filters(pd.DataFrame(rows), smoke_cfg.filters)
    assert len(out) == 3 and out["in_final_set"].tolist() == [False, True, False]


def test_cascade_adds_up(smoke_cfg):
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame([_event(event_id=str(i), station_id=f"s{i % 37}") for i in range(n)])
    df["r2"] = rng.uniform(0.5, 1, n)
    df["duration_days"] = rng.integers(2, 120, n)
    df["frac_range"] = rng.uniform(0, 1, n)
    df["veg_class"] = rng.choice(["HERB", "SAV", "WOOD", "CRO"], n)
    df["lai_m2m2"] = rng.uniform(0, 3, n)
    flagged = flag_filters(df, smoke_cfg.filters)
    steps = cascade(flagged)
    assert list(steps["after"]) == ["input", *RULES]
    assert steps["events"].iloc[0] - steps["removed"].sum() == steps["events"].iloc[-1]
    assert steps["events"].iloc[-1] == flagged["in_final_set"].sum()
    assert steps["events"].is_monotonic_decreasing and steps["stations"].is_monotonic_decreasing
    assert steps["stations"].iloc[-1] == flagged.loc[flagged.in_final_set, "station_id"].nunique()
