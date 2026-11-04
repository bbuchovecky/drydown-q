"""Config: defaults load, unknown keys and wrong types are rejected, and an
override changes only the keys it names."""

import dataclasses

import pytest

from drydown_q.config import (ConfigError, as_dict, config_hash, differences_from_default,
                              load_config, merge)
from conftest import SMOKE_PATHS


def test_defaults_load_and_match_the_baseline(smoke_cfg):
    assert smoke_cfg.events.min_decline_days == 4
    assert smoke_cfg.filters.duration_days == [6, 100]
    assert smoke_cfg.binning.ai_edges[-1] == float("inf")
    assert differences_from_default(smoke_cfg) == {}


def test_override_changes_only_its_keys(smoke_cfg):
    changed = load_config("smoke", SMOKE_PATHS, overrides={"filters": {"r2_min": 0.8}})
    assert changed.filters.r2_min == 0.8
    assert differences_from_default(changed) == {"filters.r2_min": (0.7, 0.8)}
    for section in ("prepare", "soil", "events", "fitting", "covariates", "binning"):
        assert getattr(changed, section) == getattr(smoke_cfg, section)
    other = {k: v for k, v in as_dict(changed.filters).items() if k != "r2_min"}
    assert other == {k: v for k, v in as_dict(smoke_cfg.filters).items() if k != "r2_min"}


@pytest.mark.parametrize("override, message", [
    ({"filters": {"r2_minimum": 0.8}}, "unknown key"),
    ({"filtering": {"r2_min": 0.8}}, "unknown section"),
    ({"filters": {"r2_min": "high"}}, "does not match type"),
    ({"events": {"min_decline_days": 4.5}}, "does not match type"),
    ({"events": {"min_decline_days": True}}, "does not match type"),
    ({"events": {"mask_above_fc": "yes"}}, "does not match type"),
    ({"prepare": {"grid_cell_rule": "bilinear"}}, "does not match type"),
    ({"filters": {"duration_days": [6, "100"]}}, "does not match type"),
])
def test_bad_overrides_are_rejected(override, message):
    with pytest.raises(ConfigError, match=message):
        load_config("smoke", SMOKE_PATHS, overrides=override)


def test_nullable_keys_accept_numbers_and_null():
    cfg = load_config("smoke", SMOKE_PATHS, overrides={"prepare": {"max_cell_offset_km": 10}})
    assert cfg.prepare.max_cell_offset_km == 10
    with pytest.raises(ConfigError):
        load_config("smoke", SMOKE_PATHS, overrides={"prepare": {"max_cell_offset_km": "far"}})


def test_merge_does_not_mutate_its_inputs():
    default = {"a": {"x": 1, "y": 2}}
    merged = merge(default, {"a": {"x": 5}}, "test")
    assert merged == {"a": {"x": 5, "y": 2}} and default == {"a": {"x": 1, "y": 2}}


def test_missing_run_and_paths_files_are_errors(tmp_path):
    with pytest.raises(ConfigError, match="no run file"):
        load_config("does_not_exist", SMOKE_PATHS)
    with pytest.raises(ConfigError, match="no paths file"):
        load_config("smoke", tmp_path / "nope.yml")


def test_paths_resolve_relative_to_repo(smoke_cfg):
    from drydown_q.config import REPO_ROOT
    assert smoke_cfg.paths.prepared_dir == str(REPO_ROOT / "tests" / "data" / "smoke")


def test_config_is_frozen(smoke_cfg):
    with pytest.raises(dataclasses.FrozenInstanceError):
        smoke_cfg.filters.r2_min = 0.1


def test_section_hash_changes_only_with_its_section(smoke_cfg):
    changed = load_config("smoke", SMOKE_PATHS, overrides={"filters": {"r2_min": 0.8}})
    assert config_hash(changed.filters) != config_hash(smoke_cfg.filters)
    assert config_hash(changed.fitting) == config_hash(smoke_cfg.fitting)
