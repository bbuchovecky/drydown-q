"""Prepare modules and an end-to-end run of scripts/prepare.py on tiny
synthetic raw inputs (tests/raw_fixtures.py): daily aggregation, grid cells,
forcing, MODIS, and resumability."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import raw_fixtures as fx
from conftest import REPO
from drydown_q.gridded import haversine_km, locate_cells
from drydown_q.ismn_daily import daily_means
from drydown_q.modis import daily_lai, landcover_by_year


# ── daily aggregation ─────────────────────────────────────────────────────────
def _hourly(times, values, flags):
    return pd.DataFrame({"time": pd.to_datetime(times), "theta_m3m3": values, "flag": flags})


def test_daily_means_masks_flags_and_counts_hours():
    h = _hourly(["2020-01-01 22:00", "2020-01-01 23:00", "2020-01-02 00:00",
                 "2020-01-02 01:00", "2020-01-02 02:00"],
                [0.25, 0.26, 0.30, 0.90, np.nan], ["G", "G", "G", "D01", "G"])
    d = daily_means(h, ["G"], 1)
    assert d.date.tolist() == list(pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert d.theta_m3m3.tolist() == pytest.approx([0.255, 0.30])   # 23:00 and 00:00 are different UTC days
    assert d.n_valid_hours.tolist() == [2, 1]                       # flagged and NaN records not counted


def test_daily_means_gap_days_and_min_hours():
    h = _hourly(["2020-01-01 10:00", "2020-01-01 11:00", "2020-01-04 10:00"],
                [0.2, 0.3, 0.4], ["G", "G", "G"])
    d = daily_means(h, ["G"], 1)
    assert len(d) == 4 and d.theta_m3m3.isna().tolist() == [False, True, True, False]
    assert d.n_valid_hours.tolist() == [2, 0, 0, 1]
    d2 = daily_means(h, ["G"], 2)
    assert d2.theta_m3m3.notna().tolist() == [True, False, False, False]


def test_daily_means_other_flags_option():
    h = _hourly(["2020-01-01 10:00", "2020-01-01 11:00"], [0.2, 0.4], ["G", "D01"])
    assert daily_means(h, ["G", "D01"], 1).theta_m3m3.iloc[0] == pytest.approx(0.3)


# ── grid cells ────────────────────────────────────────────────────────────────
def test_haversine_one_degree_of_latitude():
    assert haversine_km(0, 0, 1, 0, 6371.0088) == pytest.approx(111.195, rel=1e-4)


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    return fx.write_all(tmp_path_factory.mktemp("raw"))


STATIONS = pd.DataFrame({"station_id": ["NETA/St1", "NETA/St2"],
                         "lat_deg": [35.02, 35.12], "lon_deg": [-100.03, -100.27]})


def test_nearest_cell_records_offset_and_land(raw):
    first = sorted(__import__("pathlib").Path(raw["chirps_dir"]).glob("*.nc"))[0]
    cells = locate_cells(first, "precip", STATIONS, "nearest", None, 6371.0088).set_index("station_id")
    assert cells.loc["NETA/St1", "cell_is_land"] and not cells.loc["NETA/St2", "cell_is_land"]
    assert 0 < cells.loc["NETA/St1", "offset_km"] < 4


def test_nearest_land_within_km(raw, smoke_cfg):
    first = sorted(__import__("pathlib").Path(raw["chirps_dir"]).glob("*.nc"))[0]
    near = locate_cells(first, "precip", STATIONS, "nearest_land_within_km", 10, 6371.0088).set_index("station_id")
    assert near.loc["NETA/St2", "cell_is_land"]
    assert near.loc["NETA/St2", "offset_km"] <= 10
    tight = locate_cells(first, "precip", STATIONS, "nearest_land_within_km", 1, 6371.0088).set_index("station_id")
    assert not tight.loc["NETA/St2", "cell_is_land"]     # no land cell within 1 km: nearest kept, flagged
    with pytest.raises(ValueError):
        locate_cells(first, "precip", STATIONS, "nearest_land_within_km", None, 6371.0088)


# ── MODIS ─────────────────────────────────────────────────────────────────────
def test_daily_lai_drops_fill_codes_and_forward_fills(raw, smoke_cfg):
    lai = pd.read_parquet(f"{raw['appeears_dir']}/lai_terra__MOD15A2H.061.parquet")
    out = daily_lai(lai, STATIONS.station_id, smoke_cfg.prepare)
    st1 = out[out.station_id == "NETA/St1"].set_index("date")["lai_m2m2"]
    composites = pd.date_range("2019-12-27", "2020-07-31", freq="8D")
    assert st1[composites[3] + pd.Timedelta(days=5)] == pytest.approx(0.8)    # composite start, ffilled
    assert st1[composites[10]] == pytest.approx(1.4)                           # fill code 254 -> previous
    assert st1.max() < 248


def test_landcover_by_year(raw):
    lc = pd.read_parquet(f"{raw['appeears_dir']}/annual__MCD12Q1.061.parquet")
    vcf = pd.read_parquet(f"{raw['appeears_dir']}/annual__MOD44B.061.parquet")
    out = landcover_by_year(lc, vcf, STATIONS.station_id).set_index(["station_id", "year"])
    assert out.loc[("NETA/St1", 2020), "igbp_class"] == "GRA"
    assert out.loc[("NETA/St2", 2020), "igbp_class"] == "WSA"
    assert out.loc[("NETA/St1", 2019), "igbp_class"] == "CRO"
    assert out.loc[("NETA/St1", 2020), "tree_cover_pct"] == 8.0


# ── end to end ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def prepared(raw, tmp_path_factory):
    paths = tmp_path_factory.mktemp("cfg") / "paths.yml"
    paths.write_text(yaml.safe_dump(raw))
    cmd = [sys.executable, str(REPO / "scripts" / "prepare.py"), "--paths", str(paths)]
    first = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert first.returncode == 0, first.stderr[-3000:]
    second = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert second.returncode == 0, second.stderr[-3000:]
    out = __import__("pathlib").Path(raw["prepared_dir"])
    return {"sensors": pd.read_parquet(out / "sensors.parquet"),
            "day": pd.read_parquet(out / "station_day.parquet"),
            "failures": pd.read_csv(out / "failures.csv"), "log": second.stderr}


def test_prepare_sensors(prepared):
    s = prepared["sensors"].set_index("sensor_id")
    assert len(s) == 3
    sid = "NETA/St1/ProbeX_soil_moisture_0.050000_0.050000"
    assert s.loc[sid, "texture_class"] == "loam" and s.loc[sid, "sand_frac"] == pytest.approx(0.40)
    assert s.loc[sid, "chirps_cell_is_land"] and s.loc[sid, "dpet_cell_is_land"]
    assert not s.loc["NETA/St2/ProbeX_soil_moisture_0.050000_0.050000", "chirps_cell_is_land"]


def test_prepare_daily_theta(prepared):
    day = prepared["day"]
    sid = "NETA/St1/ProbeX_soil_moisture_0.050000_0.050000"
    d = day[day.sensor_id == sid].set_index("date")
    hours = pd.date_range("2020-03-15", periods=24, freq="h")
    hours = hours[hours.hour != 3]
    assert d.loc["2020-03-15", "theta_m3m3"] == pytest.approx(np.mean(fx.sm_value("St1", 0.05, hours)), abs=1e-4)
    assert d.loc["2020-03-15", "n_valid_hours"] == 23
    assert d.loc["2020-04-11", "n_valid_hours"] == 0 and np.isnan(d.loc["2020-04-11", "theta_m3m3"])
    # station_day starts ai_window_days - 1 days before the first daily value
    assert d.index.min() == pd.Timestamp("2020-03-01") - pd.Timedelta(days=59)
    assert d.loc[d.index.min(), "n_valid_hours"] == 0


def test_prepare_forcing_and_windows(prepared):
    day = prepared["day"]
    d = day[day.sensor_id == "NETA/St1/ProbeX_soil_moisture_0.050000_0.050000"].set_index("date")
    expected = fx.precip_value(pd.Timestamp("2020-03-20"), 35.0, -100.05)
    assert d.loc["2020-03-20", "precip_mm"] == pytest.approx(expected, abs=1e-4)
    window = pd.date_range(end="2020-03-20", periods=60)
    assert d.loc["2020-03-20", "precip_60d_mm"] == pytest.approx(
        sum(fx.precip_value(t, 35.0, -100.05) for t in window), rel=1e-5)
    assert np.isnan(d.loc["2020-05-05", "pet_mm"])                    # >= 100 mm/d -> missing
    assert d.loc["2020-05-06", "pet_mm"] == pytest.approx(-0.5)         # negatives kept
    # the stored window sum is complete at the first sensor day (forcing record is longer)
    assert np.isfinite(d.loc["2020-03-01", "precip_60d_mm"])
    st2 = day[day.sensor_id.str.startswith("NETA/St2")]
    assert st2.precip_mm.isna().all()                                   # ocean cell under "nearest"


def test_prepare_modis_columns(prepared):
    day = prepared["day"]
    d = day[day.sensor_id == "NETA/St1/ProbeX_soil_moisture_0.050000_0.050000"].set_index("date")
    assert d.loc["2020-03-15", "igbp_class"] == "GRA" and d.loc["2020-03-15", "tree_cover_pct"] == 8.0
    assert d.loc["2020-03-15", "lai_m2m2"] > 0


def test_prepare_rerun_skips_completed_work(prepared):
    assert prepared["failures"].empty
    assert "already prepared" in prepared["log"]


def test_covariates_recompute_the_stored_window_sums(prepared, smoke_cfg):
    from drydown_q.covariates import event_covariates
    day = prepared["day"]
    sid = "NETA/St1/ProbeX_soil_moisture_0.050000_0.050000"
    events = pd.DataFrame({"event_id": ["e1", "e2"], "sensor_id": sid,
                           "start_date": pd.to_datetime(["2020-03-01", "2020-05-20"]),
                           "end_date": pd.to_datetime(["2020-03-10", "2020-05-30"])})
    cov = event_covariates(events, day, smoke_cfg.covariates).set_index("event_id")
    d = day[day.sensor_id == sid].set_index("date")
    assert cov.loc["e1", "ai_60d"] == pytest.approx(
        d.loc["2020-03-01", "precip_60d_mm"] / d.loc["2020-03-01", "pet_60d_mm"])
    # e2's window contains the invalid dPET day (2020-05-05): not all 60 days valid -> NaN
    assert np.isnan(cov.loc["e2", "ai_60d"]) and np.isnan(d.loc["2020-05-20", "pet_60d_mm"])
