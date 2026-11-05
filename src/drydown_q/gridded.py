"""Gridded daily forcing (CHIRPS v2.0 precipitation, dPET) at stations.

Cell choice (``prepare.grid_cell_rule``):

    nearest                 the grid cell whose centre is nearest in latitude
                            and longitude (Morgan: xarray .sel(method="nearest"))
    nearest_land_within_km  the nearest cell (great-circle distance) with valid
                            data, searched within ``max_cell_offset_km``; if none,
                            the nearest cell, flagged as not land

Every station gets ``offset_km`` (great-circle distance from the station to the
centre of the cell used) and ``cell_is_land`` (that cell has a finite value on
the file's first day), whatever the rule.

Extraction reads one file at a time and writes one parquet per input file to
``out_dir``, so an interrupted run resumes where it stopped.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def window_sum(daily: pd.Series, window_days: int, min_valid_days: int) -> pd.Series:
    """Trailing sum over the ``window_days`` calendar days ending on (and
    including) each date.

        S(t) = sum_{d = t - window_days + 1}^{t} x(d)      [same unit as x]

    ``daily`` is indexed by date (one row per day; missing rows count as
    missing days). S(t) is NaN when fewer than ``min_valid_days`` of the window
    are non-missing.
    """
    daily = daily.sort_index()
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    sums = full.rolling(window_days, min_periods=min_valid_days).sum()
    return sums.reindex(daily.index)


def haversine_km(lat1, lon1, lat2, lon2, earth_radius_km: float):
    """Great-circle distance [km] on a sphere of radius ``earth_radius_km``."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat, dlon = p2 - p1, np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def _coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lat = next(n for n in ("latitude", "lat", "y") if n in ds.coords)
    lon = next(n for n in ("longitude", "lon", "x") if n in ds.coords)
    return lat, lon


def locate_cells(first_file: Path, var: str, stations: pd.DataFrame, rule: str,
                 max_km: float | None, earth_radius_km: float) -> pd.DataFrame:
    """Per station: grid ``row``, ``col``, ``offset_km``, ``cell_is_land``.

    ``stations`` has station_id, lat_deg, lon_deg. The land mask is "finite
    value on the first day of ``first_file``".
    """
    if rule == "nearest_land_within_km" and max_km is None:
        raise ValueError("grid_cell_rule = nearest_land_within_km needs prepare.max_cell_offset_km")
    with xr.open_dataset(first_file) as ds:
        lat_name, lon_name = _coord_names(ds)
        lats, lons = ds[lat_name].to_numpy(), ds[lon_name].to_numpy()
        land = np.isfinite(ds[var].isel({"time": 0}).transpose(lat_name, lon_name).to_numpy())
    # Express station longitudes in the grid's convention (-180..180 or 0..360).
    st_lon = stations["lon_deg"].to_numpy(float)
    if lons.max() > 180:
        st_lon = np.mod(st_lon, 360)
    st_lat = stations["lat_deg"].to_numpy(float)

    rows = np.abs(lats[None, :] - st_lat[:, None]).argmin(axis=1)
    cols = np.abs(lons[None, :] - st_lon[:, None]).argmin(axis=1)
    if rule == "nearest_land_within_km":
        km_per_deg = earth_radius_km * np.deg2rad(1)
        for i in np.flatnonzero(~land[rows, cols]):
            dlat = max_km / km_per_deg
            dlon = dlat / max(np.cos(np.deg2rad(st_lat[i])), np.finfo(float).eps)
            r = np.flatnonzero(np.abs(lats - st_lat[i]) <= dlat)
            c = np.flatnonzero(np.abs(lons - st_lon[i]) <= dlon)
            rr, cc = np.meshgrid(r, c, indexing="ij")
            dist = haversine_km(st_lat[i], st_lon[i], lats[rr], lons[cc], earth_radius_km)
            dist = np.where(land[rr, cc] & (dist <= max_km), dist, np.inf)
            if np.isfinite(dist).any():
                k = np.unravel_index(np.argmin(dist), dist.shape)
                rows[i], cols[i] = rr[k], cc[k]
    return pd.DataFrame({
        "station_id": stations["station_id"].to_numpy(), "row": rows, "col": cols,
        "offset_km": haversine_km(st_lat, st_lon, lats[rows], lons[cols], earth_radius_km),
        "cell_is_land": land[rows, cols],
    })


def extract_at_stations(files: list[Path], var: str, cells: pd.DataFrame, value_name: str,
                        out_dir: Path) -> pd.DataFrame:
    """Daily values of ``var`` at each station's cell, from every file in ``files``.

    Returns long ``station_id, date, <value_name>``. Each input file's result
    is cached as ``out_dir/<file stem>.parquet``; existing ones are not re-read.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unique = cells[["row", "col"]].drop_duplicates()
    parts = []
    for path in sorted(files):
        part_file = out_dir / f"{Path(path).stem}.parquet"
        if not part_file.exists():
            with xr.open_dataset(path) as ds:
                lat_name, lon_name = _coord_names(ds)
                da = ds[var].transpose("time", lat_name, lon_name)
                dates = pd.to_datetime(da["time"].to_numpy()).normalize()
                pieces = [pd.DataFrame({"row": r, "col": c, "date": dates,
                                        value_name: da[:, r, c].to_numpy().astype(float)})
                          for r, c in unique.itertuples(index=False)]
            tmp = part_file.with_suffix(".part")
            pd.concat(pieces, ignore_index=True).to_parquet(tmp, index=False)
            tmp.rename(part_file)
        parts.append(pd.read_parquet(part_file))
    values = pd.concat(parts, ignore_index=True)
    out = cells[["station_id", "row", "col"]].merge(values, on=["row", "col"])
    return out[["station_id", "date", value_name]].sort_values(["station_id", "date"],
                                                                ignore_index=True)


FORCING_COLUMNS = {"chirps": ["precip_mm", "precip_60d_mm"], "dpet": ["pet_mm", "pet_60d_mm"]}


def mask_by_cell(forcing: pd.DataFrame, cells: pd.DataFrame, max_km: float | None,
                 require_land: bool) -> tuple[pd.DataFrame, list[dict]]:
    """Set a source's forcing missing at stations whose cell is farther than
    ``max_cell_offset_km`` or (``require_land_cell``) not land.

    ``cells`` has station_id, <source>_offset_km, <source>_cell_is_land for
    source in chirps, dpet. Returns the masked forcing and one record per
    (station, source) masked. Inactive when max_km is None and not require_land.
    """
    forcing = forcing.copy()
    records = []
    for source, columns in FORCING_COLUMNS.items():
        bad = pd.Series(False, index=cells.index)
        if max_km is not None:
            bad |= cells[f"{source}_offset_km"] > max_km
        if require_land:
            bad |= ~cells[f"{source}_cell_is_land"].astype(bool)
        stations = cells.loc[bad, "station_id"]
        forcing.loc[forcing["station_id"].isin(stations), columns] = np.nan
        records += [{"id": s, "step": f"{source}_cell", "message": "cell too far or not land; forcing set missing"}
                    for s in stations]
    return forcing, records
