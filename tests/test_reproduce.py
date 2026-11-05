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
