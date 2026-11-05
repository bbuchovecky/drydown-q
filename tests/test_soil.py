"""Soil constants: texture class, Clapp & Hornberger field capacity, theta_w,
max_sm/min_sm and dz rules."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from drydown_q.soil import field_capacity, layer_depth, sensor_constants, texture_class


def test_texture_classes_and_rule_order():
    sand = pd.Series([0.40, 0.90, 0.10, 0.05, np.nan, 1.2, 0.30])
    silt = pd.Series([0.40, 0.05, 0.85, 0.50, 0.30, 0.0, 0.35])
    clay = pd.Series([0.20, 0.05, 0.05, 0.45, 0.30, 0.0, 0.35])
    out = texture_class(sand, silt, clay).tolist()
    assert out[:4] == ["loam", "sand", "silt", "silty clay"]
    assert out[4] == "unknown" and out[5] == "unknown"
    assert out[6] == "clay loam"


@pytest.mark.parametrize("texture, expected", [
    ("loam", 0.31), ("sand", 0.17), ("clay", 0.40), ("silt loam", 0.37), ("loamy sand", 0.18)])
def test_field_capacity_matches_released_morgan_values(smoke_cfg, texture, expected):
    assert field_capacity(pd.Series([texture]), smoke_cfg.soil).iloc[0] == pytest.approx(expected)


def test_field_capacity_is_nan_without_parameters(smoke_cfg):
    assert field_capacity(pd.Series(["silt", "unknown"]), smoke_cfg.soil).isna().all()


def _sensors():
    return pd.DataFrame({"sensor_id": ["a", "b", "c", "d"], "station_id": ["s1", "s1", "s1", "s2"],
                         "depth_bottom_m": [0.05, 0.20, 0.50, 0.10],
                         "texture_class": ["loam", "loam", "silt", "sand"]})


def test_layer_depth_rules(smoke_cfg):
    s = _sensors()
    assert layer_depth(s, smoke_cfg.soil).tolist() == [0.05, 0.20, 0.50, 0.10]
    thick = layer_depth(s, dataclasses.replace(smoke_cfg.soil, dz_rule="layer_thickness"))
    # station s1: boundaries 0.125, 0.35, 0.50 -> dz 0.125, 0.225, 0.15; s2 alone -> 0.10
    assert np.allclose(thick, [0.125, 0.225, 0.15, 0.10])
    fixed = layer_depth(s, dataclasses.replace(smoke_cfg.soil, dz_rule="fixed", dz_fixed_m=0.05))
    assert (fixed == 0.05).all()
    with pytest.raises(ValueError):
        layer_depth(s, dataclasses.replace(smoke_cfg.soil, dz_rule="fixed"))


def test_sensor_constants_with_fallback(smoke_cfg):
    rng = np.random.default_rng(0)
    days = pd.concat([pd.DataFrame({"sensor_id": sid, "date": pd.date_range("2020-01-01", periods=200),
                                    "theta_m3m3": rng.uniform(0.05, 0.35, 200)}) for sid in "abcd"])
    out = sensor_constants(_sensors(), days, smoke_cfg.soil).set_index("sensor_id")
    a = days[days.sensor_id == "a"].theta_m3m3
    assert out.loc["a", "min_sm_m3m3"] == a.min() == out.loc["a", "theta_w_m3m3"]
    assert out.loc["a", "theta_w_source"] == "record_min"
    assert out.loc["a", "max_sm_m3m3"] == pytest.approx(a.quantile(0.98))
    assert out.loc["a", "theta_fc_m3m3"] == 0.31 and out.loc["a", "theta_fc_source"] == "clapp_hornberger"
    assert out.loc["c", "theta_fc_source"] == "max_sm"
    assert out.loc["c", "theta_fc_m3m3"] == out.loc["c", "max_sm_m3m3"]
    q = dataclasses.replace(smoke_cfg.soil, theta_w_rule="quantile", theta_w_quantile=0.05)
    assert sensor_constants(_sensors(), days, q).set_index("sensor_id").loc["a", "theta_w_m3m3"] == \
        pytest.approx(a.quantile(0.05))


def test_seasonal_min_theta_w(smoke_cfg):
    from drydown_q.soil import wilting_point
    dates = pd.Series(pd.date_range("2018-01-01", "2020-12-31"))
    theta = pd.Series(0.20, index=dates.index)
    theta[dates.dt.year == 2018] = np.linspace(0.30, 0.10, (dates.dt.year == 2018).sum())   # min 0.10
    theta[dates.dt.year == 2019] = 0.15                                                       # min 0.15
    theta[dates.dt.year == 2020] = 0.12                                                       # min 0.12
    theta[dates == "2019-07-01"] = 0.0                     # a dropout: the record minimum
    theta[(dates.dt.year == 2020) & (dates.dt.month > 3)] = np.nan      # 2020: 91 valid days < 182
    cfg = dataclasses.replace(smoke_cfg.soil, theta_w_rule="seasonal_min")
    value, source = wilting_point(theta, dates, cfg)
    assert source == "seasonal_min" and value == pytest.approx(np.median([0.10, 0.0]))
    short = dataclasses.replace(cfg, theta_w_min_days_per_year=400)
    assert wilting_point(theta, dates, short) == (0.0, "record_min_fallback")
