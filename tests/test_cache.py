"""cached_stage computes once, reuses the result, and recomputes when its key
changes or when a previous run was interrupted."""

import pandas as pd
import pytest

from drydown_q.cache import cached_stage


def _bins():
    return {"bins": pd.DataFrame({
        "veg_class": ["HERB"], "ai_bin": [0], "lai_bin": [1], "ai_mid": [0.1],
        "lai_mid": [0.75], "n_events": [7], "q_median": [0.9], "used_in_fit": [True]})}


def test_computes_once_then_reuses(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return _bins()

    first, key1 = cached_stage("bin", compute, tmp_path, {"x": 1}, "up", "code")
    second, key2 = cached_stage("bin", compute, tmp_path, {"x": 1}, "up", "code")
    assert len(calls) == 1 and key1 == key2
    pd.testing.assert_frame_equal(first["bins"], second["bins"])


@pytest.mark.parametrize("change", [
    dict(section={"x": 2}), dict(upstream="other"), dict(code="newcode")])
def test_key_changes_with_section_upstream_and_code(tmp_path, change):
    base = dict(section={"x": 1}, upstream="up", code="code")
    _, key1 = cached_stage("bin", _bins, tmp_path, **base)
    _, key2 = cached_stage("bin", _bins, tmp_path, **{**base, **change})
    assert key1 != key2


def test_interrupted_stage_is_recomputed(tmp_path):
    _, key = cached_stage("bin", _bins, tmp_path, {"x": 1}, "up", "code")
    (tmp_path / "bin" / key / "_complete.json").unlink()
    calls = []
    cached_stage("bin", lambda: calls.append(1) or _bins(), tmp_path, {"x": 1}, "up", "code")
    assert calls == [1]


def test_schema_violation_is_raised_before_marking_complete(tmp_path):
    bad = lambda: {"bins": _bins()["bins"].drop(columns="q_median")}
    with pytest.raises(AssertionError, match="missing columns"):
        cached_stage("bin", bad, tmp_path, {"x": 1}, "up", "code")
    assert not list(tmp_path.rglob("_complete.json"))
