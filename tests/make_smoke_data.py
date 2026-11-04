"""Generate the committed synthetic 3-station dataset used by `pixi run smoke`.

Writes ``tests/data/smoke/sensors.parquet`` and ``station_day.parquet`` in the
same schemas that ``scripts/prepare.py`` produces, so ``scripts/analyze.py``
runs on them unchanged. Run with ``pixi run make-smoke-data``; the output is
deterministic (fixed seed) and committed.

This is test data, not science: the numbers below only need to produce
realistic-looking drydowns, gaps, above-field-capacity days and a spread of
aridity and LAI. Each station has one sensor and one vegetation class.

Soil water model (per day): rain adds P / (1000 dz); water above theta_fc
drains by a fixed fraction per day; below theta_star the loss follows the
q model dtheta/dt = -k ((theta - theta_w) / (theta_star - theta_w))^q, applied
with its analytical one-day solution; above theta_star the loss is k.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from drydown_q.schemas import assert_schema

OUT = Path(__file__).parent / "data" / "smoke"
SEED = 20251104
DATES = pd.date_range("2001-01-01", "2020-12-31", freq="D")

# Shared soil: loam by USDA rules (sand 0.40, silt 0.40, clay 0.20).
DEPTH_M = 0.05
POROSITY = 0.451
THETA_FC = 0.31          # matches C&H loam at -33 kPa rounded to 0.01
DRAIN_FRACTION = 0.6     # share of the water above theta_fc that drains per day
NOISE_SD = 0.003         # sensor noise, m3 m-3

STATIONS = [
    # station, igbp, tree cover, lat, lon, k, q, theta_star, theta_w, LAI min/max, wet/dry rain prob
    dict(station="SmokeGrass", igbp="GRA", tree=5.0, lat=36.5, lon=-100.2,
         k=0.045, q=0.85, theta_star=0.27, theta_w=0.04, lai=(0.3, 2.5), p_rain=(0.20, 0.04)),
    dict(station="SmokeSavanna", igbp="SAV", tree=20.0, lat=-15.1, lon=30.4,
         k=0.040, q=1.0, theta_star=0.26, theta_w=0.05, lai=(0.4, 3.2), p_rain=(0.25, 0.02)),
    dict(station="SmokeForest", igbp="ENF", tree=70.0, lat=45.2, lon=-121.8,
         k=0.035, q=1.1, theta_star=0.25, theta_w=0.06, lai=(1.5, 5.5), p_rain=(0.22, 0.06)),
]
RAIN_MEAN_MM = 9.0
PET_RANGE_MM = (1.5, 6.5)


def one_day_loss(theta, k, q, theta_star, theta_w):
    """Theta after one day of drying under the piecewise q model."""
    if theta > theta_star:
        return max(theta - k, theta_star)
    x = theta - theta_w
    if x <= 0:
        return theta
    s = theta_star - theta_w
    if abs(q - 1) < 1e-9:
        return theta_w + x * np.exp(-k / s)
    inner = x ** (1 - q) - k * (1 - q) / s ** q
    if inner <= 0:
        return theta_w
    return theta_w + inner ** (1 / (1 - q))


def simulate(st, rng, sensor_id):
    n = len(DATES)
    phase = 2 * np.pi * (DATES.day_of_year.to_numpy() - 1) / 365.25
    if st["lat"] < 0:
        phase = phase + np.pi
    season = 0.5 * (1 - np.cos(phase))               # 0 in January, 1 in July (north)
    p_wet, p_dry = st["p_rain"]
    p_rain = p_dry + (p_wet - p_dry) * (0.5 * (1 + np.cos(phase - 0.6)))
    rain = np.where(rng.random(n) < p_rain, rng.exponential(RAIN_MEAN_MM, n), 0.0)
    pet = PET_RANGE_MM[0] + (PET_RANGE_MM[1] - PET_RANGE_MM[0]) * season
    pet = pet + rng.normal(0, 0.4, n)
    lai_lo, lai_hi = st["lai"]
    lai = lai_lo + (lai_hi - lai_lo) * np.clip(season + rng.normal(0, 0.05, n), 0, 1)

    theta = np.empty(n)
    th = THETA_FC
    for i in range(n):
        th = min(th + rain[i] / (1000 * DEPTH_M), POROSITY)
        if th > THETA_FC:
            th = th - DRAIN_FRACTION * (th - THETA_FC)
        k_today = st["k"] * pet[i] / np.mean(PET_RANGE_MM)
        th = one_day_loss(th, k_today, st["q"], st["theta_star"], st["theta_w"])
        theta[i] = th
    theta = theta + rng.normal(0, NOISE_SD, n)

    # Gaps: a 10-day outage once a year, plus scattered missing days.
    missing = rng.random(n) < 0.01
    for year_start in np.flatnonzero(DATES.day_of_year == 200):
        missing[year_start:year_start + 10] = True
    theta[missing] = np.nan
    hours = np.where(missing, 0, rng.choice([24, 24, 24, 23, 12], n)).astype("int64")
    pet[rng.random(n) < 0.005] = np.nan               # occasional missing dPET

    day = pd.DataFrame({
        "sensor_id": sensor_id, "date": DATES, "theta_m3m3": theta,
        "n_valid_hours": hours, "precip_mm": rain, "pet_mm": pet,
    })
    # Trailing 60-day sums ending on (and including) each day; all 60 days required.
    day["precip_60d_mm"] = day["precip_mm"].rolling(60, min_periods=60).sum()
    day["pet_60d_mm"] = day["pet_mm"].rolling(60, min_periods=60).sum()
    # LAI as 8-day composites placed on their start date and forward-filled.
    composite = np.arange(n) // 8 * 8
    day["lai_m2m2"] = np.round(lai[composite], 1)
    day["igbp_class"] = st["igbp"]
    day["tree_cover_pct"] = st["tree"]
    return day


def main():
    rng = np.random.default_rng(SEED)
    sensors, days = [], []
    for st in STATIONS:
        sensor_id = f"SMOKE/{st['station']}/sm_{DEPTH_M:.2f}_{DEPTH_M:.2f}_synthetic"
        sensors.append(dict(
            sensor_id=sensor_id, station_id=f"SMOKE/{st['station']}", network="SMOKE",
            lat_deg=st["lat"], lon_deg=st["lon"], depth_top_m=DEPTH_M, depth_bottom_m=DEPTH_M,
            sand_frac=0.40, silt_frac=0.40, clay_frac=0.20, texture_class="loam",
            chirps_offset_km=1.2, chirps_cell_is_land=True,
            dpet_offset_km=3.4, dpet_cell_is_land=True,
        ))
        days.append(simulate(st, rng, sensor_id))
    sensors = pd.DataFrame(sensors)
    station_day = pd.concat(days, ignore_index=True)
    for c in ("sensor_id", "station_id", "network", "texture_class"):
        sensors[c] = sensors[c].astype(str)
    for c in ("sensor_id", "igbp_class"):
        station_day[c] = station_day[c].astype(str)
    assert_schema(sensors, "sensors")
    assert_schema(station_day, "station_day")
    OUT.mkdir(parents=True, exist_ok=True)
    sensors.to_parquet(OUT / "sensors.parquet", index=False)
    station_day.to_parquet(OUT / "station_day.parquet", index=False)
    print(f"wrote {len(sensors)} sensors, {len(station_day)} station-days to {OUT}")


if __name__ == "__main__":
    main()
