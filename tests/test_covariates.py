"""Covariates: trailing window sums, AI_60 at the start day, LAI statistics
and the vegetation class mapping."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from drydown_q.covariates import assign_class, event_covariates
from drydown_q.gridded import window_sum


def test_window_sum_includes_end_day_and_needs_all_days():
    s = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=10))
    out = window_sum(s, 3, 3)
    assert np.isnan(out.iloc[1]) and out.iloc[2] == 3.0
    s.iloc[5] = np.nan
    out = window_sum(s, 3, 3)
    assert out.iloc[5:8].isna().all() and out.iloc[8] == 3.0


def test_window_sum_treats_missing_rows_as_missing_days():
    idx = pd.date_range("2020-01-01", periods=10).delete(4)
    out = window_sum(pd.Series(1.0, index=idx), 3, 3)
    assert out.loc["2020-01-06":"2020-01-07"].isna().all()
    assert out.loc["2020-01-08"] == 3.0


def _station_day(n=100):
    dates = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "sensor_id": "s", "date": dates, "precip_mm": 1.0, "pet_mm": 2.0,
        "lai_m2m2": np.arange(n, dtype=float), "igbp_class": "GRA", "tree_cover_pct": 5.0})


def _events(start, end):
    return pd.DataFrame({"event_id": ["e"], "sensor_id": ["s"],
                         "start_date": [pd.Timestamp(start)], "end_date": [pd.Timestamp(end)]})


def test_ai_at_start_day_with_full_window(smoke_cfg):
    day = _station_day()
    day.loc[day.date == "2020-03-10", "precip_mm"] = 61.0   # start day = day 69 (0-based)
    cov = event_covariates(_events("2020-03-10", "2020-03-20"), day, smoke_cfg.covariates).iloc[0]
    assert cov.ai_60d == pytest.approx((59 + 61) / 120)
    assert cov.lai_m2m2 == 69.0 and cov.igbp_class == "GRA"

    excl = dataclasses.replace(smoke_cfg.covariates, ai_window_includes_start=False)
    cov = event_covariates(_events("2020-03-10", "2020-03-20"), day, excl).iloc[0]
    assert cov.ai_60d == pytest.approx(60 / 120)


def test_ai_is_nan_without_a_full_window(smoke_cfg):
    cov = event_covariates(_events("2020-01-30", "2020-02-05"), _station_day(), smoke_cfg.covariates)
    assert np.isnan(cov.ai_60d.iloc[0])


@pytest.mark.parametrize("statistic, expected", [("event_max", 79.0), ("event_mean", 74.0)])
def test_lai_statistics(smoke_cfg, statistic, expected):
    cfg = dataclasses.replace(smoke_cfg.covariates, lai_statistic=statistic)
    cov = event_covariates(_events("2020-03-10", "2020-03-20"), _station_day(), cfg)
    assert cov.lai_m2m2.iloc[0] == expected


def test_assign_class_morgan_3():
    out = assign_class(pd.Series(["GRA", "SAV", "WSA", "OSH", "CRO", "XXX"]), "morgan_3")
    assert out.iloc[:5].tolist() == ["HERB", "SAV", "WOOD", "SHR", "CRO"] and pd.isna(out.iloc[5])
    with pytest.raises(ValueError):
        assign_class(pd.Series(["GRA"]), "no_such_scheme")
