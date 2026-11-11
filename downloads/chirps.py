"""CHIRPS v2.0 daily precipitation at the ISMN station cells, from the official CHC files.

    pixi run download-chirps
    python downloads/chirps.py [--run morgan2025] [--paths config/paths.yml]
                               [--settings config/downloads/chirps.yml] [--years 2026 ...]

The Climate Hazards Center serves CHIRPS v2.0 only as global annual netCDF files
(about 1.1 GiB each) and offers no access to single cells. So each needed year is
downloaded, the cells prepare.py would use are kept, and the file is deleted
before the next one: disk use peaks at one annual file, and the kept extract is a
few MB per year.

Which cells. The run's ``prepare.grid_cell_rule`` (with ``max_cell_offset_km``
and ``earth_radius_km``) applied to the stations of prepare step 1 by the same
function prepare.py uses (``gridded.locate_cells``), so the station values
equal those read from the full files. Stations sharing a cell download it once.
Under the ``nearest`` rule, stations beyond the grid's latitude range (CHIRPS
covers 50 S - 50 N) get the grid's edge cell, as in Morgan's code; the distance
is recorded in ``offset_km``.

Which years. Every year that overlaps a station's record extended back by
2 x (covariates.ai_window_days - 1) days (station_day starts
ai_window_days - 1 days before a record, and its trailing window sums need as
many days again), and that has a file in the server's directory listing.
Record years without a file (before 1981, when CHIRPS starts) are reported.

Writes to ``paths.chirps_dir`` (which prepare.py then reads):

    grid.npz                              cell-centre lat/lon and land mask (first day of the first file downloaded)
    cells.parquet                         station_id, row, col, cell_lat_deg, cell_lon_deg, offset_km, cell_is_land
    chirps-v2.0.<year>.days_p05.parquet   row, col, date, precip_mm at the station cells
    download_log.csv                      one row per year downloaded: status, bytes, days, last date, message

A year whose table already holds every needed cell is skipped, so an interrupted
run resumes and a changed station set only fetches what is missing. ``--years``
re-downloads the years listed (e.g. the current year, which CHC extends monthly).
Needs prepare step 1 (``pixi run prepare`` runs it and stops while the forcing
paths are unset). Exit status 1 if any year failed.
"""

import argparse
import logging
import time
import urllib.error
from pathlib import Path

import pandas as pd

from drydown_q import download, gridded, ismn_daily
from drydown_q.config import CONFIG_DIR, ChirpsDownload, load_config, load_download

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--run", default="morgan2025", help="run file whose `prepare` and `covariates` settings apply")
parser.add_argument("--paths", default=None, help="paths file (default config/paths.yml)")
parser.add_argument("--settings", default=CONFIG_DIR / "downloads" / "chirps.yml", help="download settings file")
parser.add_argument("--years", type=int, nargs="+", default=None,
                    help="download only these years, replacing their tables")
args = parser.parse_args()
started = time.time()

cfg = load_config(args.run, args.paths)
s = load_download(args.settings, ChirpsDownload)
p, cov = cfg.prepare, cfg.covariates
if cfg.paths.chirps_dir is None:
    raise SystemExit("set chirps_dir in the paths file: the CHIRPS extract is written there")
out = Path(cfg.paths.chirps_dir)
incoming = out / "incoming"          # one annual file at a time, deleted after extraction
out.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=cfg.runtime.log_level, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    force=True, handlers=[logging.StreamHandler(),
                                          logging.FileHandler(out / "download.log", mode="a")])
log = logging.getLogger("download_chirps")
fetch_args = (s.timeout_s, s.retries, s.retry_wait_s, s.read_chunk_mb << 20)

# ── 1. Stations and the years they need ───────────────────────────────────────
# The stations are prepare.py's: sensors of prepare step 1 (prepare.sensor_unit)
# that have a daily file, including those whose file is empty.
work = Path(cfg.paths.prepared_dir) / "work"
sensors_file = work / "ismn_sensors.parquet"
if not sensors_file.exists():
    raise SystemExit(f"no {sensors_file}: run prepare step 1 first (pixi run prepare)")
sensors, _ = ismn_daily.sensor_units(pd.read_parquet(sensors_file), p.sensor_unit)
daily_dir = work / ismn_daily.daily_dir_name(p)
sensors["daily_file"] = [daily_dir / f"{ismn_daily.file_key(sid)}.parquet" for sid in sensors.sensor_id]
sensors = sensors[[f.exists() for f in sensors["daily_file"]]].reset_index(drop=True)
stations = sensors.drop_duplicates("station_id")[["station_id", "lat_deg", "lon_deg"]].reset_index(drop=True)
spans = ismn_daily.record_spans(sensors)
lead = pd.Timedelta(days=2 * (cov.ai_window_days - 1))
record_years = sorted({year for r in spans.itertuples()
                       for year in range((r.first - lead).year, r.last.year + 1)})
# Only years with a file on the server can be fetched (CHIRPS starts in 1981; the
# current year's file grows monthly). Record days in other years get no precipitation.
listing = set(download.list_directory(s.base_url, s.timeout_s))
needed_years = [y for y in record_years if s.file_pattern.format(year=y) in listing]
no_file = sorted(set(record_years) - set(needed_years))
years = sorted(set(args.years) & set(needed_years)) if args.years else needed_years
if not needed_years or (args.years and not years):
    raise SystemExit(f"none of the requested years has a file at {s.base_url}")
names = {year: s.file_pattern.format(year=year) for year in needed_years}
tables = {year: out / f"{Path(name).stem}.parquet" for year, name in names.items()}
log.info("1. %d stations; records (with the window lead) span %d-%d; %d of those years have a CHIRPS file "
         "(%d-%d), %d do not; this run: %s", len(stations), record_years[0], record_years[-1],
         len(needed_years), needed_years[0], needed_years[-1], len(no_file),
         ", ".join(map(str, years)) if args.years else "all years with a file")

# ── 2. Grid and station cells ─────────────────────────────────────────────────
# The grid (cell centres, land mask) comes from the first file downloaded and is
# kept, so the cells can be chosen again later without any download.
grid_file = out / gridded.GRID_FILE
if not grid_file.exists():
    first = incoming / names[years[0]]
    if not first.exists():
        log.info("2. downloading %s for the grid", names[years[0]])
        download.fetch(f"{s.base_url}/{names[years[0]]}", first, *fetch_args)
    gridded.save_grid(grid_file, *gridded.read_grid(first, s.variable))
lats, lons, land = gridded.load_grid(grid_file)
cells = gridded.locate_cells(lats, lons, land, stations, p.grid_cell_rule, p.max_cell_offset_km,
                             p.earth_radius_km)
cells.assign(cell_lat_deg=lats[cells["row"]], cell_lon_deg=lons[cells["col"]]).to_parquet(
    out / "cells.parquet", index=False)
needed_cells = set(cells[["row", "col"]].itertuples(index=False, name=None))
outside = (stations["lat_deg"] < lats.min()) | (stations["lat_deg"] > lats.max())
log.info("2. %d stations -> %d cells (%s rule); %d stations beyond the grid's latitudes use an edge cell "
         "(largest offset %.0f km); %d stations' cells are not land", len(cells), len(needed_cells),
         p.grid_cell_rule, outside.sum(), cells["offset_km"].max(), (~cells["cell_is_land"]).sum())

# ── 3. One annual file at a time: download, keep the station cells, delete ────
log_rows = []
for year in years:
    if not args.years and tables[year].exists():
        have = set(pd.read_parquet(tables[year], columns=["row", "col"]).itertuples(index=False, name=None))
        if needed_cells <= have:
            log.info("3. %d: already extracted", year)
            continue
    nc = incoming / names[year]
    row = {"time": pd.Timestamp.now().isoformat(timespec="seconds"), "year": year,
           "url": f"{s.base_url}/{names[year]}", "status": "ok", "message": ""}
    try:
        t0 = time.time()
        if not nc.exists():
            download.fetch(row["url"], nc, *fetch_args)
        row["bytes"] = nc.stat().st_size
        values = gridded.extract_cells(nc, s.variable, cells, "precip_mm")
        values.to_parquet(tables[year].with_suffix(".part"), index=False)
        tables[year].with_suffix(".part").rename(tables[year])
        nc.unlink()
        row.update(n_days=values["date"].nunique(), last_date=values["date"].max().date())
        log.info("3. %d: %.0f MiB file, fetched and extracted in %.0f s -> %d days x %d cells (%.1f MiB kept)", year,
                 row["bytes"] / (1 << 20), time.time() - t0, row["n_days"], len(needed_cells),
                 tables[year].stat().st_size / (1 << 20))
    except urllib.error.HTTPError as error:         # e.g. 404: the year is not on the server
        row.update(status="failed", message=f"HTTP {error.code} {error.reason}")
    except OSError as error:                        # network failure after all retries, or an unreadable file
        nc.unlink(missing_ok=True)
        row.update(status="failed", message=str(error))
    if row["status"] == "failed":
        log.warning("3. %d: FAILED: %s", year, row["message"])
    log_rows.append(row)

log_file = out / "download_log.csv"
if log_rows:
    pd.DataFrame(log_rows).to_csv(log_file, mode="a", header=not log_file.exists(), index=False)

# ── 4. Coverage ───────────────────────────────────────────────────────────────
# Missing precipitation counts as dry in event detection (events.precip_missing_is_dry),
# so name every needed year without a table and any ISMN days outside CHIRPS.
done = [y for y in needed_years if tables[y].exists()]
missing = sorted(set(needed_years) - set(done))
if missing:
    log.warning("4. years with a CHIRPS file but not yet extracted: %s", ", ".join(map(str, missing)))
if no_file:
    before = spans[spans["first"].dt.year < needed_years[0]]
    log.warning("4. ISMN records cover %d years with no CHIRPS file (%s); %d stations have days before "
                "%d, which will have no precipitation", len(no_file), ", ".join(map(str, no_file)),
                len(before), needed_years[0])
if done:
    chirps_end = pd.read_parquet(tables[max(done)], columns=["date"])["date"].max()
    after = spans[spans["last"] > chirps_end]
    if len(after):
        log.warning("4. CHIRPS ends %s; %d stations have ISMN days after it (latest %s), which will have "
                    "no precipitation", chirps_end.date(), len(after), after["last"].max().date())
log.info("done in %.0f s: %d of %d needed years extracted in %s", time.time() - started,
         len(done), len(needed_years), out)
raise SystemExit(1 if any(r["status"] == "failed" for r in log_rows) else 0)
