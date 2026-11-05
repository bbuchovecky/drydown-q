"""Tier 2 event matching (report.compare_events) on toy tables."""

import pandas as pd
import pytest

from drydown_q.report import compare_events


def test_compare_events_shares_and_differences():
    ours = pd.DataFrame({"sensor_id": ["a", "a", "b", "c"],
                         "start_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-05", "2020-01-01"]),
                         "q": [1.0, 1.2, 0.8, 2.0]})
    released = pd.DataFrame({"sensor_id": ["a", "b", "b", "d"],
                             "start_date": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-03-01", "2020-01-01"]),
                             "q_q": [1.1, 0.8, 0.9, 1.0]})
    summary, matched = compare_events(ours, released, {"q": "q_q"})
    s = summary.set_index("quantity")["value"]
    assert s["matched events"] == 2
    assert s["share of released events reproduced"] == 0.5
    assert s["share of our events that are new"] == 0.5
    assert s["sensors in both"] == 2
    assert s["share reproduced, sensors in both"] == 2 / 3
    assert s["share new, sensors in both"] == 1 - 2 / 3
    assert s["median |difference| q"] == pytest.approx(0.05)
    assert len(matched) == 2


def test_match_sensors_by_name_then_unique_depth():
    from drydown_q.report import match_sensors
    ours = pd.DataFrame({"sensor_id": ["N/S/a", "N/S/new1", "N/S/new2", "N/S/new3"], "network": "N",
                         "station": "S", "sensor": ["a", "new1", "new2", "new3"],
                         "depth_top_m": [0.0, 0.1, 0.2, 0.2], "depth_bottom_m": [0.05, 0.1, 0.2, 0.2]})
    theirs = pd.DataFrame({"network": "N", "station": "S", "sensor": ["a", "old1", "old2", "old3"],
                           "depth_top_m": [0.0, 0.1, 0.2, 0.2], "depth_bottom_m": [0.05, 0.1, 0.2, 0.2]})
    m = match_sensors(ours, theirs).set_index("our_sensor_id")
    assert m.loc["N/S/a", "match"] == "name"
    assert m.loc["N/S/new1", "their_sensor"] == "old1" and m.loc["N/S/new1", "match"] == "depth"
    assert "N/S/new2" not in m.index          # two sensors at 0.2 m on both sides: ambiguous
