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
    return sensors


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


def daily_means(hourly: pd.DataFrame, flags_kept: list[str], min_valid_hours: int) -> pd.DataFrame:
    """UTC daily mean theta of the records whose ISMN flag is in ``flags_kept``.

        theta_day = mean(theta_h : flag_h in flags_kept, theta_h not NaN)   [m3 m-3]
        n_valid_hours = number of those records

    ISMN harmonised data are hourly, so the count is in hours. Days with fewer
    than ``min_valid_hours`` kept records are NaN (Morgan: no minimum, i.e. 1).
    The result has one row for every day from the first to the last kept
    record; days without kept records are NaN with n_valid_hours = 0.
    """
    kept = hourly[hourly["flag"].isin(flags_kept) & hourly["theta_m3m3"].notna()]
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


def write_daily_files(archive: Path, meta_path: Path, sensors: pd.DataFrame,
                      flags_kept: list[str], min_valid_hours: int, out_dir: Path) -> list[dict]:
    """Write ``out_dir/<file_key>.parquet`` (date, theta_m3m3, n_valid_hours) for
    each sensor in ``sensors`` that does not have one yet. One call handles a
    batch of sensors (one joblib task); a sensor that fails is recorded in the
    returned list and the batch continues."""
    out_dir = Path(out_dir)
    todo = sensors[[not (out_dir / f"{file_key(s)}.parquet").exists() for s in sensors["sensor_id"]]]
    if todo.empty:
        return []
    ds = open_archive(archive, meta_path)
    failures = []
    for s in todo.itertuples():
        try:
            daily = daily_means(read_hourly(ds, s.network, s.station, s.sensor),
                                flags_kept, min_valid_hours)
        except (OSError, ValueError, KeyError, IndexError) as err:
            failures.append({"id": s.sensor_id, "step": "ismn_daily",
                             "message": f"{type(err).__name__}: {err}"})
            continue
        tmp = out_dir / f"{file_key(s.sensor_id)}.part"
        daily.to_parquet(tmp, index=False)
        tmp.rename(out_dir / f"{file_key(s.sensor_id)}.parquet")
    return failures
