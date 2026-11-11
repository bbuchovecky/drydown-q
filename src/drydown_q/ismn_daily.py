"""ISMN in situ soil moisture: sensor metadata and UTC daily means.

The archive is read with the ``ismn`` package (ISMN_Interface). Its metadata
cache is written to ``meta_path`` (under outputs/), never into the archive.
Units are converted here, once: ISMN texture fractions are % weight -> [0-1];
soil moisture is already m3 m-3.
"""

import hashlib
from pathlib import Path

import pandas as pd
from ismn.interface import ISMN_Interface

from drydown_q.soil import texture_class

VARIABLE = "soil_moisture"
PERCENT = 100   # unit conversion: % weight -> fraction


def open_archive(archive: Path, meta_path: Path) -> ISMN_Interface:
    return ISMN_Interface(str(archive), meta_path=str(meta_path), parallel=False)


def sensor_id(network: str, station: str, sensor_name: str) -> str:
    """``network/station/<ismn sensor name>``, e.g.
    ``SCAN/Abrams/Hydraprobe-Analog_soil_moisture_0.050800_0.050800``."""
    return f"{network}/{station}/{sensor_name}"


def file_key(sid: str) -> str:
    """Filesystem-safe short key for per-sensor intermediate files."""
    return hashlib.sha1(sid.encode()).hexdigest()[:16]


def read_sensors(ds: ISMN_Interface) -> pd.DataFrame:
    """One row per soil moisture sensor: ids, coordinates, depth range [m], and
    the texture fractions ISMN assigns to the sensor's depth (static variables,
    e.g. HWSD), converted to fractions, with the USDA texture class."""
    rows = []
    for net, stat, sens in ds.collection.iter_sensors(variable=VARIABLE):
        md = sens.metadata
        rows.append({
            "sensor_id": sensor_id(net.name, stat.name, sens.name),
            "station_id": f"{net.name}/{stat.name}", "network": net.name,
            "station": stat.name, "sensor": sens.name, "instrument": sens.instrument,
            "lat_deg": float(stat.lat), "lon_deg": float(stat.lon),
            "depth_top_m": float(sens.depth.start), "depth_bottom_m": float(sens.depth.end),
            "sand_frac": _value(md, "sand_fraction") / PERCENT,
            "silt_frac": _value(md, "silt_fraction") / PERCENT,
            "clay_frac": _value(md, "clay_fraction") / PERCENT,
        })
    sensors = pd.DataFrame(rows)
    sensors["texture_class"] = texture_class(sensors["sand_frac"], sensors["silt_frac"],
                                             sensors["clay_frac"])
    parts = sensors["instrument"].str.extract(INSTRUMENT_PATTERN)
    return pd.concat([sensors, parts], axis=1)


# Recent ISMN archives name instruments "<model>_<probe>_<period>": one file per
# deployment period of a probe (e.g. Stevens-Hydraprobe-Analog_1_3).
INSTRUMENT_PATTERN = r"^(?P<instrument_model>.+)_(?P<probe_index>\d+)_(?P<period_index>\d+)$"


def sensor_units(files: pd.DataFrame, unit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Define the analysis sensors from the archive's files (``prepare.sensor_unit``).

    ``file``: every ISMN file is a sensor.
    ``probe``: the period files of one probe (same network, station, instrument
        model, probe index and depth range) are one sensor, named
        ``<model>_<probe>_soil_moisture_<depth_from>_<depth_to>``. A change of
        instrument model starts a new sensor. Files whose instrument name has no
        probe/period suffix stay single-file sensors.

    Returns ``(sensors, members)``: one row per sensor (metadata of its first
    file), and the long table ``sensor_id, network, station, sensor, period``
    listing its files in period order.
    """
    files = files.copy()
    if "instrument_model" not in files:
        files = pd.concat([files, files["instrument"].str.extract(INSTRUMENT_PATTERN)], axis=1)
    files["period"] = pd.to_numeric(files["period_index"], errors="coerce")
    if unit == "file":
        files["unit_name"] = files["sensor"]
    else:
        split = files["instrument_model"].notna()
        files["unit_name"] = files["sensor"]
        files.loc[split, "unit_name"] = [
            f"{m}_{i}_{VARIABLE}_{top:.6f}_{bottom:.6f}" for m, i, top, bottom in
            files.loc[split, ["instrument_model", "probe_index", "depth_top_m", "depth_bottom_m"]].itertuples(index=False)]
    files["unit_id"] = files["network"] + "/" + files["station"] + "/" + files["unit_name"]
    files = files.sort_values(["unit_id", "period"])
    members = files[["unit_id", "network", "station", "sensor", "period"]].rename(columns={"unit_id": "sensor_id"})
    first = files.drop_duplicates("unit_id")
    sensors = first.assign(sensor_id=first["unit_id"], sensor=first["unit_name"])
    sensors = sensors.drop(columns=["unit_id", "unit_name", "period"]).reset_index(drop=True)
    return sensors, members.reset_index(drop=True)


def _value(md, key: str) -> float:
    try:
        return float(md[key].val)
    except (KeyError, TypeError, ValueError):
        return float("nan")


def read_hourly(ds: ISMN_Interface, network: str, station: str, sensor_name: str) -> pd.DataFrame:
    """Raw records of one sensor: ``time`` (UTC), ``theta_m3m3``, ``flag``."""
    sens = ds[network][station][sensor_name]
    data = sens.read_data()
    return pd.DataFrame({"time": pd.to_datetime(data.index), "theta_m3m3": data[VARIABLE].to_numpy(float),
                         "flag": data[f"{VARIABLE}_flag"].astype(str).to_numpy()})


# ISMN soil moisture flags for frozen conditions (ISMN_qualityflags_description.txt):
# D01 in situ soil temperature < 0 C, D02 in situ air temperature < 0 C, D03 GLDAS
# soil temperature < 0 C.
FROZEN_FLAGS = {"D01", "D02", "D03"}


def kept_records(flags: pd.Series, flags_kept: list[str], exclude_frozen: bool) -> pd.Series:
    """True for records whose every ISMN flag (records can carry several, e.g.
    "D01,D02") is in ``flags_kept``; with ``exclude_frozen`` (prepare.exclude_snow_frozen),
    records carrying any of FROZEN_FLAGS are dropped even if those flags are kept."""
    parts = flags.astype(str).str.split(",")
    keep = parts.map(lambda f: all(x in flags_kept for x in f))
    if exclude_frozen:
        keep &= ~parts.map(lambda f: any(x in FROZEN_FLAGS for x in f))
    return keep.astype(bool)


def daily_means(hourly: pd.DataFrame, flags_kept: list[str], min_valid_hours: int,
                exclude_frozen: bool) -> pd.DataFrame:
    """UTC daily mean theta of the kept records (``kept_records``).

        theta_day = mean(theta_h : flag_h in flags_kept, theta_h not NaN)   [m3 m-3]
        n_valid_hours = number of those records

    ISMN harmonised data are hourly, so the count is in hours. Days with fewer
    than ``min_valid_hours`` kept records are NaN (Morgan: no minimum, i.e. 1).
    The result has one row for every day from the first to the last kept
    record; days without kept records are NaN with n_valid_hours = 0.
    """
    kept = hourly[kept_records(hourly["flag"], flags_kept, exclude_frozen) & hourly["theta_m3m3"].notna()]
    if kept.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                             "theta_m3m3": pd.Series(dtype=float),
                             "n_valid_hours": pd.Series(dtype="int64")})
    day = kept["time"].dt.floor("D")
    g = kept.groupby(day)["theta_m3m3"]
    daily = pd.DataFrame({"theta_m3m3": g.mean(), "n_valid_hours": g.size()})
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    daily["n_valid_hours"] = daily["n_valid_hours"].fillna(0).astype("int64")
    daily.loc[daily["n_valid_hours"] < min_valid_hours, "theta_m3m3"] = float("nan")
    return daily.rename_axis("date").reset_index()


def write_daily_files(archive: Path, meta_path: Path, sensors: pd.DataFrame, members: pd.DataFrame,
                      flags_kept: list[str], min_valid_hours: int, exclude_frozen: bool,
                      out_dir: Path) -> list[dict]:
    """Write ``out_dir/<file_key>.parquet`` (date, theta_m3m3, n_valid_hours) for
    each sensor in ``sensors`` that does not have one yet. A sensor's files
    (``members``) are read in period order and joined; consecutive period files
    of a probe do not overlap, and at a timestamp present in two files the first
    record is kept. One call handles a batch of sensors (one joblib task); a
    sensor that fails is recorded in the returned list and the batch continues."""
    out_dir = Path(out_dir)
    todo = sensors[[not (out_dir / f"{file_key(s)}.parquet").exists() for s in sensors["sensor_id"]]]
    if todo.empty:
        return []
    ds = open_archive(archive, meta_path)
    failures = []
    files_of = {sid: g for sid, g in members[members["sensor_id"].isin(todo["sensor_id"])].groupby("sensor_id")}
    for s in todo.itertuples():
        try:
            parts = [read_hourly(ds, f.network, f.station, f.sensor) for f in files_of[s.sensor_id].itertuples()]
            hourly = pd.concat(parts, ignore_index=True).drop_duplicates("time", keep="first")
            daily = daily_means(hourly, flags_kept, min_valid_hours, exclude_frozen)
        except (OSError, ValueError, KeyError, IndexError) as err:
            failures.append({"id": s.sensor_id, "step": "ismn_daily",
                             "message": f"{type(err).__name__}: {err}"})
            continue
        tmp = out_dir / f"{file_key(s.sensor_id)}.part"
        daily.to_parquet(tmp, index=False)
        tmp.rename(out_dir / f"{file_key(s.sensor_id)}.parquet")
    return failures


def insitu_precip(ds: ISMN_Interface, flags_kept: list[str]) -> pd.DataFrame:
    """Daily in situ precipitation per station: ``station_id, date, precip_insitu_mm``.

        P_day = sum(P_h : flag_h in flags_kept)      [mm d-1], UTC days

    A day with no kept record is NaN (not 0). A partly observed day is the sum
    of its kept records. Each file is summed on its own; files of one station
    (deployment periods, which do not overlap, or replicate gauges) are then
    averaged day by day.
    """
    daily = []
    for net, stat, sens in ds.collection.iter_sensors(variable="precipitation"):
        data = sens.read_data()
        kept = data[kept_records(data["precipitation_flag"], flags_kept, False)]["precipitation"].astype(float)
        if kept.empty:
            continue
        sums = kept.groupby(pd.to_datetime(kept.index).floor("D")).sum(min_count=1)
        daily.append(pd.DataFrame({"station_id": f"{net.name}/{stat.name}", "date": sums.index,
                                   "precip_insitu_mm": sums.to_numpy(float)}))
    if not daily:
        return pd.DataFrame({"station_id": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                             "precip_insitu_mm": pd.Series(dtype=float)})
    out = pd.concat(daily, ignore_index=True)
    return out.groupby(["station_id", "date"], as_index=False)["precip_insitu_mm"].mean()


def record_spans(sensors: pd.DataFrame) -> pd.DataFrame:
    """First and last date of the daily files of each station: ``station_id, first, last``.

    ``sensors`` has station_id and daily_file; stations whose files are empty are left out.
    """
    spans = []
    for station_id, group in sensors.groupby("station_id"):
        dates = pd.concat([pd.read_parquet(f, columns=["date"])["date"] for f in group["daily_file"]])
        if not dates.empty:
            spans.append({"station_id": station_id, "first": dates.min(), "last": dates.max()})
    return pd.DataFrame(spans, columns=["station_id", "first", "last"])


def daily_dir_name(cfg) -> str:
    """Folder (under prepared_dir/work) of the per-sensor daily files for the
    settings that determine them, so changing one of them never reuses stale files."""
    from drydown_q.config import config_hash
    return f"daily_{cfg.sensor_unit}_" + config_hash(cfg.ismn_flags_kept, cfg.min_valid_hours,
                                                     cfg.exclude_snow_frozen)
