"""Tiny synthetic raw inputs in the real formats, for testing prepare.py.

- ISMN archive (header_values .stm files + static_variables.csv), 2 stations:
  NETA/St1 (sensors at 0.05 and 0.20 m) and NETA/St2 (one sensor at 0.05 m,
  on a coastal cell whose nearest CHIRPS cell is ocean).
- CHIRPS-like and dPET-like yearly netCDF files on small grids.
- AppEEARS tidy parquet files (appeears_pipeline format).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

STATIONS = {"St1": (35.02, -100.03), "St2": (35.12, -100.27)}
YEARS = [2019, 2020]
STATIC = """quantity_name;unit;depth_from[m];depth_to[m];value;description;quantity_source_name;quantity_source_description;quantity_source_provider;quantity_source_version;quantity_source_resolution;quantity_source_timerange;quantity_source_url;
saturation;m^3*m^-3;0.00;0.30;0.41;;HWSD;x;x;v1.1;30";None;x;
clay fraction;% weight;0.00;0.30;20.00;;HWSD;x;x;v1.1;30";None;x;
sand fraction;% weight;0.00;0.30;40.00;;HWSD;x;x;v1.1;30";None;x;
silt fraction;% weight;0.00;0.30;40.00;;HWSD;x;x;v1.1;30";None;x;
clay fraction;% weight;0.30;1.00;30.00;;HWSD;x;x;v1.1;30";None;x;
sand fraction;% weight;0.30;1.00;30.00;;HWSD;x;x;v1.1;30";None;x;
silt fraction;% weight;0.30;1.00;40.00;;HWSD;x;x;v1.1;30";None;x;
land cover classification;;;;120;Shrubland;CCI_landcover_2010;x;ESA;x;300m;x;x;
climate classification;;;;BSk;Arid;koeppen_geiger_2007;x;None;x;0.1;None;x;
"""


def sm_value(station, depth, t):
    """Deterministic hourly soil moisture, so tests can recompute daily means."""
    base = 0.20 if station == "St1" else 0.15
    return base + 0.05 * np.sin(2 * np.pi * t.day_of_year / 365) - 0.02 * depth + 0.001 * t.hour


def write_ismn(root: Path) -> Path:
    archive = root / "ismn"
    times = pd.date_range("2020-03-01", "2020-06-30 23:00", freq="h")
    sensors = [("St1", 0.05), ("St1", 0.20), ("St2", 0.05)]
    for station, (lat, lon) in STATIONS.items():
        folder = archive / "NETA" / station
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"NETA_NETA_{station}_static_variables.csv").write_text(STATIC)
    for station, depth in sensors:
        lat, lon = STATIONS[station]
        values = sm_value(station, depth, times)
        flags = np.where(times.hour == 3, "D01", "G")              # 03 UTC is always flagged bad
        keep = ~((times >= "2020-04-10") & (times < "2020-04-13"))   # a 3-day outage
        header = f"NETA NETA {station} {lat:.5f} {lon:.5f} 500.0 {depth:.4f} {depth:.4f} 'ProbeX'\n"
        lines = [f"{t:%Y/%m/%d %H:%M} {v:.4f} {f} M" for t, v, f, k in zip(times, values, flags, keep) if k]
        name = f"NETA_NETA_{station}_sm_{depth:.6f}_{depth:.6f}_ProbeX_20200301_20200630.stm"
        (archive / "NETA" / station / name).write_text(header + "\n".join(lines) + "\n")
    return archive


def _grid(lat0, lat1, lon0, lon1, step):
    return np.round(np.arange(lat0, lat1, step), 4), np.round(np.arange(lon0, lon1, step), 4)


def precip_value(date, lat, lon):
    return float((date.day_of_year % 7) + 10 * abs(lat) % 3 + abs(lon) % 2)


def write_grids(root: Path) -> tuple[Path, Path]:
    chirps, dpet = root / "chirps", root / "dpet"
    chirps.mkdir(parents=True, exist_ok=True)
    dpet.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        lats, lons = _grid(34.9, 35.3, -100.4, -99.9, 0.05)
        p = np.array([[[precip_value(d, la, lo) for lo in lons] for la in lats] for d in days])
        p[:, :, lons < -100.22] = np.nan       # western cells are "ocean"
        xr.Dataset({"precip": (("time", "latitude", "longitude"), p.astype("float32"))},
                   coords={"time": days, "latitude": lats, "longitude": lons}
                   ).to_netcdf(chirps / f"chirps-v2.0.{year}.days_p05.nc")
        lats, lons = _grid(34.85, 35.4, -100.45, -99.8, 0.1)
        e = np.broadcast_to((3 + 2 * np.sin(2 * np.pi * days.day_of_year.to_numpy() / 365))[:, None, None],
                            (len(days), len(lats), len(lons))).copy()
        e[days == f"{year}-05-05"] = 150.0          # invalid (>= pet_valid_max_mm_day)
        e[days == f"{year}-05-06"] = -0.5           # negative, kept
        xr.Dataset({"pet": (("time", "lat", "lon"), e.astype("float32"))},
                   coords={"time": days, "lat": lats, "lon": lons}
                   ).to_netcdf(dpet / f"{year}_daily_pet.nc")
    return chirps, dpet


def sid(station):
    return hashlib.sha1(f"NETA/{station}".encode()).hexdigest()[:10]


def write_appeears(root: Path) -> Path:
    out = root / "appeears"
    out.mkdir(parents=True, exist_ok=True)
    composites = pd.date_range("2019-12-27", "2020-07-31", freq="8D")
    rows = []
    for station in STATIONS:
        for i, d in enumerate(composites):
            raw = 254.0 if i == 10 else round(0.5 + 0.1 * i, 1)   # one fill code (water)
            for di, dj in [(0, 0), (1, 0)]:
                rows.append({"sid": sid(station), "di": di, "dj": dj, "date": d,
                             "Lai_500m": raw if raw < 248 else np.nan,
                             "Lai_500m_raw": raw if di == 0 else 9.9})
    pd.DataFrame(rows).to_parquet(out / "lai_terra__MOD15A2H.061.parquet", index=False)
    lc, vcf = [], []
    for station, igbp in [("St1", 10), ("St2", 8)]:
        for year in YEARS:
            lc.append({"sid": sid(station), "di": 0, "dj": 0, "date": pd.Timestamp(f"{year}-01-01"),
                       "LC_Type1": igbp if year == 2020 else 12})
            vcf.append({"sid": sid(station), "di": 0, "dj": 0, "date": pd.Timestamp(f"{year}-03-06"),
                        "Percent_Tree_Cover": 7.0 + year - 2019})
    pd.DataFrame(lc).to_parquet(out / "annual__MCD12Q1.061.parquet", index=False)
    pd.DataFrame(vcf).to_parquet(out / "annual__MOD44B.061.parquet", index=False)
    return out


def write_all(root: Path) -> dict:
    archive = write_ismn(root)
    chirps, dpet = write_grids(root)
    return {"ismn_archive": str(archive), "chirps_dir": str(chirps), "dpet_dir": str(dpet),
            "appeears_dir": str(write_appeears(root)), "morgan_repo": None,
            "outputs_dir": str(root / "outputs"), "prepared_dir": str(root / "prepared")}
