"""Binning: left-closed edges, minimum counts, midpoint lookup by bin code
(also when bins are empty), and the surface fit."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from drydown_q.binning import bin_medians, fit_surface, midpoints, surface, surface_set


def _events(rows):
    return pd.DataFrame(rows, columns=["veg_class", "ai_60d", "lai_m2m2", "q"])


def test_midpoints_with_open_last_bin(smoke_cfg):
    b = smoke_cfg.binning
    assert np.allclose(midpoints(b.ai_edges, b.open_bin_upper_for_mid["ai"]),
                       [0.1, 0.35, 0.575, 0.725, 0.9, 1.25, 1.75, 2.6])
    assert np.allclose(midpoints(b.lai_edges, b.open_bin_upper_for_mid["lai"]),
                       [0.25, 0.75, 1.5, 2.5, 4, 6])


def test_edges_are_left_closed(smoke_cfg):
    # AI exactly 0.5 belongs to [0.5, 0.65) (code 2); LAI exactly 1 to [1, 2) (code 2).
    bins = bin_medians(_events([("HERB", 0.5, 1.0, 0.9)]), ["HERB"], smoke_cfg.binning)
    hit = bins[bins.n_events == 1].iloc[0]
    assert (hit.ai_bin, hit.lai_bin, hit.ai_mid, hit.lai_mid) == (2, 2, 0.575, 1.5)


def test_right_closed_option(smoke_cfg):
    cfg = dataclasses.replace(smoke_cfg.binning, bins_right_closed=True)
    bins = bin_medians(_events([("HERB", 0.5, 1.0, 0.9)]), ["HERB"], cfg)
    hit = bins[bins.n_events == 1].iloc[0]
    assert (hit.ai_bin, hit.lai_bin) == (1, 1)


def test_midpoint_lookup_by_code_when_most_bins_are_empty(smoke_cfg):
    # Only the last AI bin and a middle LAI bin are occupied: midpoints must
    # come from the bin codes (2.6, 4.0), not from order of appearance (0.1, 0.25).
    events = _events([("WOOD", 5.0, 4.0, q) for q in [1.0, 1.1, 1.2, 1.3, 1.4]])
    bins = bin_medians(events, ["WOOD"], smoke_cfg.binning)
    assert len(bins) == 8 * 6
    hit = bins[bins.n_events > 0]
    assert len(hit) == 1
    assert (hit.ai_mid.item(), hit.lai_mid.item(), hit.q_median.item()) == (2.6, 4.0, 1.2)


def test_min_events_per_bin(smoke_cfg):
    four = _events([("HERB", 0.1, 0.25, 1.0)] * 4)
    five = _events([("HERB", 0.1, 0.25, 1.0)] * 5)
    assert not bin_medians(four, ["HERB"], smoke_cfg.binning)["used_in_fit"].any()
    assert bin_medians(five, ["HERB"], smoke_cfg.binning)["used_in_fit"].sum() == 1


def test_missing_q_or_covariate_is_not_counted(smoke_cfg):
    events = _events([("HERB", 0.1, 0.25, np.nan), ("HERB", np.nan, 0.25, 1.0), ("HERB", 0.1, 0.25, 1.0)])
    assert bin_medians(events, ["HERB"], smoke_cfg.binning)["n_events"].sum() == 1


def test_surface_recovers_known_plane(smoke_cfg):
    a, b, c, d = 0.65, 0.02, -0.13, 0.55
    rows = []
    for ai in [0.1, 0.35, 0.575, 0.9, 1.75]:
        for lai in [0.75, 1.5, 2.5, 4.0]:
            rows += [("HERB", ai, lai, surface((ai, lai), a, b, c, d))] * 5
    bins = bin_medians(_events(rows), ["HERB"], smoke_cfg.binning)
    fit = fit_surface(bins, ["HERB"], smoke_cfg.binning).iloc[0]
    assert fit.fit_status == "ok" and fit.n_bins == 20 and fit.n_events == 100
    assert np.allclose([fit.a, fit.b, fit.c, fit.d], [a, b, c, d], atol=1e-8)
    assert fit.r2_bins == pytest.approx(1.0)


def test_too_few_bins_is_recorded_not_fatal(smoke_cfg):
    bins = bin_medians(_events([("SAV", 0.1, 0.25, 1.0)] * 5), ["SAV"], smoke_cfg.binning)
    fit = fit_surface(bins, ["SAV"], smoke_cfg.binning).iloc[0]
    assert fit.fit_status == "too_few_bins" and np.isnan(fit.a)


def test_surface_set_uses_strict_lai(smoke_cfg):
    a = pd.DataFrame({"in_final_set": [True, True, False], "lai_m2m2": [0.5, 0.6, 3.0]})
    mask = surface_set(a, smoke_cfg.filters.lai_min_m2m2, smoke_cfg.binning.lai_min_inclusive_surface)
    assert mask.tolist() == [False, True, False]
