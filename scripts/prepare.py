"""Raw data -> the `sensors` and `station_day` tables (slow; run once).

    pixi run prepare
    python scripts/prepare.py [--run morgan2025] [--paths config/paths.yml] [--force]

Steps (each skips work that is already done, so an interrupted run resumes):

1. ISMN archive (``paths.ismn_archive``) with the ``ismn`` reader: soil moisture
   sensors and their metadata; per sensor, UTC daily means of the records
   flagged ``prepare.ismn_flags_kept``, with ``n_valid_hours``.
2. CHIRPS v2.0 0.05 deg precipitation (``paths.chirps_dir``) and dPET 0.1 deg
   (``paths.dpet_dir``) at each station's cell (``prepare.grid_cell_rule``),
   recording the station-to-cell offset and whether the cell is land. dPET
   values >= ``pet_valid_max_mm_day`` become NaN. Trailing window sums of P and
   PET (``covariates.ai_window_days``, ``ai_window_min_valid_days``) are taken on
   the full forcing record, so they are complete at the start of a sensor record.
3. MODIS from the tidy AppEEARS outputs (``paths.appeears_dir``): daily LAI
   (``prepare.lai_product``), and IGBP class and tree cover per year.
4. Assemble and write ``sensors.parquet`` and ``station_day.parquet`` to
   ``paths.prepared_dir``. station_day starts ``ai_window_days - 1`` days before
   each sensor's first daily value, so analysis can recompute the window sums.

A record of every failed sensor or station is written to ``failures.csv``.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed, effective_n_jobs

from drydown_q import gridded, ismn_daily, modis
from drydown_q.config import config_hash, load_config, write_resolved
from drydown_q.schemas import SCHEMAS, assert_schema

# Input file layouts (data formats, not science).
CHIRPS_GLOB, CHIRPS_VAR = "chirps-v2.0.*.days_p05.nc", "precip"
DPET_GLOB, DPET_VAR = "*_daily_pet.nc", "pet"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--run", default="morgan2025", help="run file whose `prepare` settings apply")
parser.add_argument("--paths", default=None, help="paths file (default config/paths.yml)")
parser.add_argument("--force", action="store_true",
                    help="allow reuse of a prepared_dir made with different settings")
args = parser.parse_args()
started = time.time()

cfg = load_config(args.run, args.paths)
p, cov = cfg.prepare, cfg.covariates
prepared = Path(cfg.paths.prepared_dir)
work = prepared / "work"
work.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=cfg.runtime.log_level, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    force=True, handlers=[logging.StreamHandler(),
                                          logging.FileHandler(prepared / "prepare.log", mode="a")])
log = logging.getLogger("prepare")

# The prepared tables depend on the `prepare` section and the AI window.
key = config_hash(p, {"ai_window_days": cov.ai_window_days,
                      "ai_window_min_valid_days": cov.ai_window_min_valid_days})
manifest_file = prepared / "prepare_manifest.json"
if manifest_file.exists() and json.loads(manifest_file.read_text())["key"] != key and not args.force:
    raise SystemExit(f"{prepared} was prepared with other settings; use another prepared_dir or --force")
if manifest_file.exists() and (prepared / "station_day.parquet").exists() and not args.force:
    log.info("%s is already prepared with these settings; nothing to do", prepared)
    raise SystemExit(0)
write_resolved(cfg, prepared / "config.resolved.yml")
failures = []

# ── 1. ISMN: sensors and UTC daily means ──────────────────────────────────────
if cfg.paths.ismn_archive is None:
    raise SystemExit("paths.ismn_archive is not set in the paths file")
archive, meta_path = Path(cfg.paths.ismn_archive), work / "ismn_meta"
sensors_file = work / "ismn_sensors.parquet"
if not sensors_file.exists():
    ismn_daily.read_sensors(ismn_daily.open_archive(archive, meta_path)).to_parquet(sensors_file, index=False)
files = pd.read_parquet(sensors_file)
# The analysis sensor is a whole probe record (prepare.sensor_unit = probe) or one archive file.
sensors, members = ismn_daily.sensor_units(files, p.sensor_unit)
log.info("1. ISMN: %d soil moisture files -> %d sensors (unit = %s) at %d stations",
         len(files), len(sensors), p.sensor_unit, sensors.station_id.nunique())

daily_dir = work / ismn_daily.daily_dir_name(p)
daily_dir.mkdir(exist_ok=True)
n_batches = max(1, min(len(sensors), effective_n_jobs(cfg.runtime.n_jobs)))
batches = [sensors.iloc[rows] for rows in np.array_split(np.arange(len(sensors)), n_batches)]
results = Parallel(n_jobs=cfg.runtime.n_jobs)(
    delayed(ismn_daily.write_daily_files)(archive, meta_path, b, members, p.ismn_flags_kept,
                                          p.min_valid_hours, p.exclude_snow_frozen, daily_dir)
    for b in batches)
failures += [f for batch in results for f in batch]
sensors["daily_file"] = [daily_dir / f"{ismn_daily.file_key(s)}.parquet" for s in sensors.sensor_id]
sensors = sensors[[f.exists() for f in sensors["daily_file"]]].reset_index(drop=True)
log.info("   daily means written for %d sensors (%d failed)", len(sensors), len(failures))

# In situ precipitation (used only by events.precip_source = insitu_then_chirps).
insitu_file = work / f"insitu_precip_{config_hash(p.ismn_flags_kept)}.parquet"
if not insitu_file.exists():
    ismn_daily.insitu_precip(ismn_daily.open_archive(archive, meta_path), p.ismn_flags_kept).to_parquet(
        insitu_file, index=False)
insitu = pd.read_parquet(insitu_file)
log.info("   in situ precipitation at %d stations", insitu.station_id.nunique())

# Steps 2-4 need the gridded forcing and MODIS inputs; without them, stop here
# (the ISMN work above is kept, and a later run resumes from it).
missing_inputs = [n for n in ("chirps_dir", "dpet_dir", "appeears_dir") if getattr(cfg.paths, n) is None]
if missing_inputs:
    pd.DataFrame(failures, columns=["id", "step", "message"]).to_csv(prepared / "failures.csv", index=False)
    log.info("stopping after step 1: paths %s not set", ", ".join(missing_inputs))
    raise SystemExit(0)

# Date span of each station's daily records (forcing is kept from window start).
spans = []
for station_id, group in sensors.groupby("station_id"):
    dates = pd.concat([pd.read_parquet(f, columns=["date"])["date"] for f in group["daily_file"]])
    if not dates.empty:
        spans.append({"station_id": station_id, "first": dates.min(), "last": dates.max()})
spans = pd.DataFrame(spans)
stations = sensors.drop_duplicates("station_id")[["station_id", "lat_deg", "lon_deg"]]

# ── 2. CHIRPS and dPET at the stations ────────────────────────────────────────
forcing_file = work / "forcing.parquet"
cells_file = work / "cells.parquet"
if not forcing_file.exists():
    forcing, cells = [], []
    for source, folder, glob, var, column in [("chirps", cfg.paths.chirps_dir, CHIRPS_GLOB, CHIRPS_VAR, "precip_mm"),
                                              ("dpet", cfg.paths.dpet_dir, DPET_GLOB, DPET_VAR, "pet_mm")]:
        files = sorted(Path(folder).glob(glob))
        if not files:
            raise SystemExit(f"no {glob} files in {folder}")
        loc = gridded.locate_cells(*gridded.read_grid(files[0], var), stations, p.grid_cell_rule,
                                   p.max_cell_offset_km, p.earth_radius_km)
        values = gridded.extract_at_stations(files, var, loc, column,
                                             work / f"{source}_{config_hash(loc[['row', 'col']].to_dict('list'))}")
        cells.append(loc[["station_id", "offset_km", "cell_is_land"]].add_prefix(f"{source}_")
                     .rename(columns={f"{source}_station_id": "station_id"}))
        forcing.append(values.set_index(["station_id", "date"]))
    forcing = pd.concat(forcing, axis=1).reset_index()
    # dPET quality rule: values >= pet_valid_max_mm_day are missing; negatives are kept (Morgan).
    forcing.loc[forcing["pet_mm"] >= p.pet_valid_max_mm_day, "pet_mm"] = np.nan
    parts = []
    for station_id, f in forcing.groupby("station_id"):
        f = f.set_index("date").sort_index()
        f["precip_60d_mm"] = gridded.window_sum(f["precip_mm"], cov.ai_window_days, cov.ai_window_min_valid_days)
        f["pet_60d_mm"] = gridded.window_sum(f["pet_mm"], cov.ai_window_days, cov.ai_window_min_valid_days)
        span = spans[spans.station_id == station_id]
        if span.empty:
            continue
        lo = span["first"].iloc[0] - pd.Timedelta(days=cov.ai_window_days - 1)
        parts.append(f.loc[lo:span["last"].iloc[0]].reset_index())
    pd.concat(parts, ignore_index=True).to_parquet(forcing_file, index=False)
    cells[0].merge(cells[1], on="station_id").to_parquet(cells_file, index=False)
forcing = pd.read_parquet(forcing_file)
cells = pd.read_parquet(cells_file)
log.info("2. forcing: %d station-days; median offset CHIRPS %.2f km, dPET %.2f km; non-land cells %d / %d",
         len(forcing), cells.chirps_offset_km.median(), cells.dpet_offset_km.median(),
         (~cells.chirps_cell_is_land).sum(), (~cells.dpet_cell_is_land).sum())

# Optional exclusions (max_cell_offset_km, require_land_cell): forcing from a
# distant or non-land cell becomes missing, and the station is listed in failures.
forcing, masked = gridded.mask_by_cell(forcing, cells, p.max_cell_offset_km, p.require_land_cell)
failures += masked
if masked:
    log.info("   forcing set missing for %d (station, source) pairs by the cell rule", len(masked))

# ── 3. MODIS LAI, land cover and tree cover ───────────────────────────────────
modis_file = work / "lai_daily.parquet"
lc_file = work / "landcover.parquet"
if not modis_file.exists() or not lc_file.exists():
    appeears = Path(cfg.paths.appeears_dir)
    modis.daily_lai(pd.read_parquet(modis.product_file(appeears, p.lai_product)),
                    stations.station_id, p).to_parquet(modis_file, index=False)
    modis.landcover_by_year(pd.read_parquet(modis.product_file(appeears, "MCD12Q1.061")),
                            pd.read_parquet(modis.product_file(appeears, "MOD44B.061")),
                            stations.station_id).to_parquet(lc_file, index=False)
lai = pd.read_parquet(modis_file)
landcover = pd.read_parquet(lc_file)
log.info("3. MODIS: LAI at %d stations, land cover at %d stations",
         lai.station_id.nunique(), landcover.station_id.nunique())

# ── 4. Assemble sensors and station_day ───────────────────────────────────────
out_sensors = sensors.merge(cells, on="station_id", how="left")
columns = list(SCHEMAS["sensors"]["required"])
out_sensors = out_sensors[columns]
assert_schema(out_sensors, "sensors")
out_sensors.to_parquet(prepared / "sensors.parquet.part", index=False)

day_columns = list(SCHEMAS["station_day"]["required"])
writer, n_rows = None, 0
tmp_day = prepared / "station_day.parquet.part"
for station_id, group in sensors.groupby("station_id", sort=True):
    f_st = forcing[forcing.station_id == station_id].drop(columns="station_id")
    lai_st = lai[lai.station_id == station_id].drop(columns="station_id")
    lc_st = landcover[landcover.station_id == station_id].drop(columns="station_id")
    p_st = insitu[insitu.station_id == station_id].drop(columns="station_id")
    for s in group.itertuples():
        d = pd.read_parquet(s.daily_file)
        if d.empty:
            continue
        pad = pd.date_range(d.date.min() - pd.Timedelta(days=cov.ai_window_days - 1), d.date.max(), freq="D")
        d = d.set_index("date").reindex(pad).rename_axis("date").reset_index()
        d["n_valid_hours"] = d["n_valid_hours"].fillna(0).astype("int64")
        d = (d.merge(f_st, on="date", how="left").merge(lai_st, on="date", how="left")
             .merge(p_st, on="date", how="left")
             .assign(year=lambda x: x.date.dt.year).merge(lc_st, on="year", how="left"))
        d["sensor_id"] = s.sensor_id
        d["igbp_class"] = d["igbp_class"].astype(str).where(d["igbp_class"].notna(), "")
        for c in ["theta_m3m3", "precip_mm", "pet_mm", "precip_60d_mm", "pet_60d_mm",
                  "lai_m2m2", "tree_cover_pct", "precip_insitu_mm"]:
            d[c] = d[c].astype(float)
        d["sensor_id"] = d["sensor_id"].astype(str)
        d["date"] = d["date"].astype("datetime64[ns]")
        d = d[day_columns + ["precip_insitu_mm"]]
        assert_schema(d, "station_day")
        table = pa.Table.from_pandas(d, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_day, table.schema)
        writer.write_table(table.cast(writer.schema))
        n_rows += len(d)
if writer is not None:
    writer.close()
(prepared / "sensors.parquet.part").rename(prepared / "sensors.parquet")
tmp_day.rename(prepared / "station_day.parquet")
pd.DataFrame(failures, columns=["id", "step", "message"]).to_csv(prepared / "failures.csv", index=False)
manifest_file.write_text(json.dumps({"key": key, "sensors": len(out_sensors), "station_days": n_rows,
                                     "failures": len(failures),
                                     "wall_seconds": round(time.time() - started, 1)}, indent=2))
log.info("4. wrote %d sensors and %d sensor-days to %s (%d failures recorded)",
         len(out_sensors), n_rows, prepared, len(failures))
