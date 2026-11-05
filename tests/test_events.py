"""Event detection on hand-written series with known starts and ends.

Every series uses dz = 0.05 m, so tau = 0.5 mm/d / (1000 * 0.05 m) = 0.01
m3 m-3 d-1, and theta_fc = 0.31.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from drydown_q.events import detect_sensor, find_events, find_starts

THETA_FC, DZ = 0.31, 0.05


def _day(theta, precip=None):
    n = len(theta)
    return pd.DataFrame({"date": pd.date_range("2020-06-01", periods=n),
                         "theta_m3m3": np.asarray(theta, dtype=float),
                         "precip_mm": np.zeros(n) if precip is None else np.asarray(precip, float),
                         "pet_mm": np.full(n, 4.0)})


# No difference sits exactly on tau: 0.19 - 0.20 is -0.010000000000000009 in floating point.
# day:   0     1     2     3     4     5     6     7      8     9      10     11
CLEAN = [0.20, 0.20, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.195, 0.19, 0.185, 0.185]
# Delta: 0   +0.10 -0.02 -0.02 -0.02 -0.02 -0.02 -0.005 -0.005 -0.005  0   (last NaN)


def _one(theta, cfg, precip=None):
    ev = detect_sensor(_day(theta, precip), THETA_FC, DZ, cfg)
    return ev


def test_clean_drydown(smoke_cfg):
    ev = _one(CLEAN, smoke_cfg.events)
    assert len(ev) == 1
    # Candidates are days 2-6 (Delta < -0.01) -> start 2. End: first Delta > -0.01 -> day 7.
    assert (ev.start[0], ev.end[0], ev.end_reason[0]) == (2, 7, "decline_below_threshold")
    assert ev.n_obs[0] == 6 and ev.n_days_above_fc[0] == 0


def test_rain_inside_start_window_moves_or_removes_the_start(smoke_cfg):
    rain_day3 = np.zeros(len(CLEAN)); rain_day3[3] = 5.0
    assert _one(CLEAN, smoke_cfg.events, rain_day3).empty            # only days 4-6 left: 3 < 4
    rain_day2 = np.zeros(len(CLEAN)); rain_day2[2] = 5.0
    longer = CLEAN[:7] + [0.20, 0.18, 0.16, 0.16, 0.16]               # candidates 3..8
    ev = _one(longer, smoke_cfg.events, rain_day2)
    assert ev.start.tolist() == [3]
    at_threshold = rain_day2.copy(); at_threshold[2] = 2.0             # P <= 2 mm is still dry
    assert _one(longer, smoke_cfg.events, at_threshold).start.tolist() == [2]


def test_rain_after_the_start_window_does_not_end_the_event(smoke_cfg):
    rain = np.zeros(len(CLEAN)); rain[6] = 10.0
    ev = _one(CLEAN, smoke_cfg.events, rain)
    assert (ev.start[0], ev.end[0]) == (2, 7)


def test_missing_precip_counts_as_dry_unless_configured(smoke_cfg):
    rain = np.zeros(len(CLEAN)); rain[3] = np.nan
    assert _one(CLEAN, smoke_cfg.events, rain).start.tolist() == [2]
    strict = dataclasses.replace(smoke_cfg.events, precip_missing_is_dry=False)
    assert _one(CLEAN, strict, rain).empty


def test_data_gap_ends_the_event(smoke_cfg):
    gappy = list(CLEAN); gappy[6] = np.nan; gappy[7] = np.nan
    # Back-fill: days 6, 7 become 0.195 (day 8). Delta_5 = 0.195 - 0.24 < -tau (continues),
    # Delta_6 = 0 -> the event ends on day 6, the first gap day.
    ev = _one(gappy, smoke_cfg.events)
    assert (ev.start[0], ev.end[0], ev.end_reason[0]) == (2, 6, "gap")
    assert ev.n_obs[0] == 4


def test_values_above_field_capacity_are_masked(smoke_cfg):
    wet = [0.20, 0.40, 0.36, 0.33, 0.30, 0.28, 0.26, 0.24, 0.22, 0.215, 0.21, 0.21]
    ev = _one(wet, smoke_cfg.events)
    # Masked: days 1-3 (> theta_fc) take day 4's 0.30, so Delta_3 = 0 and the first
    # candidate is day 4.
    assert ev.start.tolist() == [4] and ev.n_days_above_fc[0] == 0
    unmasked = dataclasses.replace(smoke_cfg.events, mask_above_fc=False)
    ev = _one(wet, unmasked)
    assert ev.start.tolist() == [1] and ev.n_days_above_fc[0] == 3


def test_record_ending_mid_event_gives_duration_zero(smoke_cfg):
    tail = [0.20, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20]                # still declining at the end
    ev = _one(tail, smoke_cfg.events)
    assert (ev.start[0], ev.end[0], ev.end_reason[0]) == (1, 1, "record_end")


def test_separate_end_threshold(smoke_cfg):
    cfg = dataclasses.replace(smoke_cfg.events, end_min_decline_mm_day=1.5)   # tau_end = 0.03
    ev = _one(CLEAN, cfg)
    assert (ev.start[0], ev.end[0]) == (2, 2)          # Delta_2 = -0.02 > -0.03: ends at once


def test_runs_and_overlaps():
    c = pd.Series([0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    assert find_starts(c, 4) == [1, 7]
    assert find_starts(c, 5) == [1]
    assert find_starts(pd.Series([True, True]), 4) == []


def test_find_events_tables(smoke_cfg):
    day = _day(CLEAN + CLEAN).assign(sensor_id="s")
    day = day.drop(index=[22])                                        # a missing row is a missing day
    constants = pd.DataFrame({"sensor_id": ["s"], "theta_fc_m3m3": [THETA_FC], "dz_m": [DZ]})
    out = find_events(day, constants, smoke_cfg.events)
    ev, days = out["events"], out["event_days"]
    assert ev.event_id.tolist() == ["s/2020-06-03", "s/2020-06-15"]
    assert ev.duration_days.tolist() == [5, 5]
    assert days.groupby("event_id").t_days.max().tolist() == [5, 5]
    assert days[days.event_id == "s/2020-06-15"].theta_m3m3.isna().sum() == 0
    from drydown_q.schemas import assert_schema
    assert_schema(ev, "events")
    assert_schema(days, "event_days")


def test_rain_ends_the_event_with_whole_event_check(smoke_cfg):
    rain = np.zeros(len(CLEAN)); rain[6] = 10.0                     # after the start window (days 2-5)
    cfg = dataclasses.replace(smoke_cfg.events, precip_check="whole_event")
    ev = _one(CLEAN, cfg, rain)
    assert (ev.start[0], ev.end[0], ev.end_reason[0]) == (2, 6, "rain")
    assert _one(CLEAN, smoke_cfg.events, rain).end.tolist() == [7]     # start_window: rain ignored


def test_precip_source_insitu_then_chirps(smoke_cfg):
    day = _day(CLEAN).assign(sensor_id="s")
    day["precip_insitu_mm"] = 0.0
    day.loc[3, "precip_insitu_mm"] = 6.0            # the gauge saw rain on day 3; CHIRPS did not
    day.loc[0, "precip_mm"] = 6.0                   # CHIRPS saw rain on day 0: equal wet-day shares
    constants = pd.DataFrame({"sensor_id": ["s"], "theta_fc_m3m3": [THETA_FC], "dz_m": [DZ]})
    assert len(find_events(day, constants, smoke_cfg.events)["events"]) == 1          # chirps only
    cfg = dataclasses.replace(smoke_cfg.events, precip_source="insitu_then_chirps")
    assert find_events(day, constants, cfg)["events"].empty                             # gauge rain breaks the start
    with pytest.raises(ValueError, match="precip_insitu_mm"):
        find_events(day.drop(columns="precip_insitu_mm"), constants, cfg)


def test_gauge_quality_checks(smoke_cfg):
    from drydown_q.events import choose_precip
    cfg = dataclasses.replace(smoke_cfg.events, precip_source="insitu_then_chirps")
    day = pd.concat([_day(CLEAN).assign(sensor_id=sid) for sid in ["OK/s/x", "ICN/s/x", "WET/s/x"]],
                    ignore_index=True)
    day["precip_insitu_mm"] = 0.0
    day.loc[day.date == day.date.iloc[3], "precip_insitu_mm"] = 6.0
    day.loc[day.date == day.date.iloc[0], "precip_mm"] = 6.0
    day.loc[day.sensor_id == "WET/s/x", "precip_insitu_mm"] = 6.0      # gauge wet every day: 12x CHIRPS
    out, sources = choose_precip(day, cfg)
    src = sources.set_index("sensor_id")
    assert src.loc["OK/s/x", "source"] == "insitu_then_chirps"
    assert src.loc["ICN/s/x", "source"] == "chirps" and "excluded" in src.loc["ICN/s/x", "reason"]
    assert src.loc["WET/s/x", "source"] == "chirps" and "wet-day share" in src.loc["WET/s/x", "reason"]
    assert out[out.sensor_id == "OK/s/x"].precip_mm.iloc[3] == 6.0
    assert out[out.sensor_id == "WET/s/x"].precip_mm.iloc[3] == 0.0
