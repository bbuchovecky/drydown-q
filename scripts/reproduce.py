"""Compare our pipeline with the released results of Morgan et al. (2025).

Tier 1 (downstream only): the released table
``<morgan_repo>/data/ismn_results_star_th0_04may.csv.zip`` stands in for our
``fits`` table, and analysis steps 4-9 (covariates, filters, final set, bins,
surfaces) run on it with the baseline config. Every number is compared with
``data/tables/morgan2025_tier1_targets.csv``; a mismatch is a bug in our code
until shown otherwise. Exit status 1 if any check fails.

Tier 2 (full rebuild): our events and fits from ``analyze.py --run morgan2025``
(steps 0-3 from raw ISMN data) are matched with the released events on
(sensor, start date). Reported: the share of released events reproduced, the
share of our events that are new, and the median |difference| in q, k and
theta_star over matched events. Settings are never tuned to force a match.

    pixi run reproduce                      # tier 1
    python scripts/reproduce.py --tier 2    # after `pixi run analyze --run morgan2025`
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from drydown_q.binning import bin_medians, fit_surface, surface_set
from drydown_q.cache import read_stage
from drydown_q.config import TABLES_DIR, load_config, write_resolved
from drydown_q.covariates import add_derived, assign_class
from drydown_q.filters import QUALITY_RULES, cascade, flag_filters
from drydown_q.fitting import fit_event
from drydown_q.ismn_daily import PERCENT as ismn_percent, file_key, sensor_units
from drydown_q.report import compare_events, markdown_table, match_sensors, write_report
from drydown_q.soil import sensor_constants

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--paths", default=None, help="paths file (default config/paths.yml)")
parser.add_argument("--refit-sample", type=int, default=0,
                    help="released events refitted with our fit_event (0 = skip)")
parser.add_argument("--tier", type=int, choices=[1, 2], default=1)
parser.add_argument("--check-ismn", action="store_true",
                    help="compare prepare step 1 (sensors, daily theta) with Morgan's metadata and events")
parser.add_argument("--run", default="morgan2025", help="tier 2: the analyze run to compare")
args = parser.parse_args()

cfg = load_config(args.run if args.tier == 2 else "morgan2025", args.paths)
run_dir = Path(cfg.paths.outputs_dir) / "runs" / ("check_ismn" if args.check_ismn else f"reproduce_tier{args.tier}")
run_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=cfg.runtime.log_level, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S", force=True,
                    handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "reproduce.log", mode="w")])
log = logging.getLogger("reproduce")
write_resolved(cfg, run_dir / "config.resolved.yml")
if cfg.paths.morgan_repo is None:
    raise SystemExit("paths.morgan_repo is not set: reproduce needs the released Morgan et al. data")
data = Path(cfg.paths.morgan_repo) / "data"
ours = []   # (check, veg_class, quantity, value)

# ── ISMN check: prepare step 1 against Morgan's sensors and event theta ──────
# Needs only the ISMN part of prepare.py. Sensors are paired by exact name, else
# by unique (network, station, depth) because instrument names change between
# archive versions. Daily theta is compared on every day of every released
# event of a paired sensor (the released `theta` arrays are unmasked daily means).
if args.check_ismn:
    work = Path(cfg.paths.prepared_dir) / "work"
    ours, _ = sensor_units(pd.read_parquet(work / "ismn_sensors.parquet"), cfg.prepare.sensor_unit)
    daily_dir = work / f"daily_{cfg.prepare.sensor_unit}"
    meta = pd.read_csv(data / "ISMN" / "metadata.csv")
    meta = meta[meta["variable"] == "soil_moisture"].rename(
        columns={"depth_from": "depth_top_m", "depth_to": "depth_bottom_m"})
    pairs = match_sensors(ours, meta[["network", "station", "sensor", "depth_top_m", "depth_bottom_m"]])
    released = pd.read_csv(data / "ismn_results_star_th0_04may.csv.zip",
                           usecols=["SITE_ID", "Sensor_ID", "event_start", "time", "theta",
                                    "min_sm", "max_sm", "theta_fc"])
    released = released.merge(meta[["station", "sensor", "network", "depth_top_m", "depth_bottom_m"]],
                              left_on=["SITE_ID", "Sensor_ID"], right_on=["station", "sensor"],
                              validate="many_to_one")

    # Released event-day theta, one row per (sensor, day).
    long = []
    for r in released.itertuples():
        t = np.array(r.time.strip("[]").split(), dtype=float).astype(int)
        long.append(pd.DataFrame({"network": r.network, "their_station": r.station, "their_sensor": r.sensor,
                                  "date": pd.Timestamp(r.event_start) + pd.to_timedelta(t, unit="D"),
                                  "released": np.array(r.theta.strip("[]").split(), dtype=float)}))
    long = pd.concat(long, ignore_index=True)
    daily_cache = {}

    def our_daily(sid):
        if sid not in daily_cache:
            daily_cache[sid] = pd.read_parquet(daily_dir / f"{file_key(sid)}.parquet")
        return daily_cache[sid]

    def equal_share(sid, rel_days):
        joined = rel_days.merge(our_daily(sid), on="date", how="left")
        return np.isclose(joined["theta_m3m3"], joined["released"], rtol=0,
                          atol=np.finfo(np.float32).eps).mean()

    # Replicate probes at one depth: pair by data, choosing the candidate whose
    # daily theta equals the released event values on the most days (unique best only).
    rel_sensors = released[["network", "station", "sensor", "depth_top_m", "depth_bottom_m"]].drop_duplicates()
    rel_sensors = rel_sensors.rename(columns={"station": "their_station", "sensor": "their_sensor"})
    paired = rel_sensors.merge(pairs, on=["network", "their_station", "their_sensor"])
    todo = rel_sensors[~rel_sensors["their_sensor"].isin(paired["their_sensor"])
                       | ~rel_sensors["their_station"].isin(paired["their_station"])]
    todo = todo.merge(paired[["network", "their_station", "their_sensor"]], how="left", indicator=True)
    todo = todo[todo["_merge"] == "left_only"].drop(columns="_merge")
    used = set(pairs["our_sensor_id"])
    by_data = []
    ours_by_depth = {k: g["sensor_id"].tolist() for k, g in ours.groupby(
        [ours["network"], ours["station"].str.lower(), ours["depth_top_m"], ours["depth_bottom_m"]])}
    rel_by_sensor = {k: g[["date", "released"]] for k, g in long.groupby(["network", "their_station", "their_sensor"])}
    for r in todo.itertuples():
        cands = [sid for sid in ours_by_depth.get((r.network, r.their_station.lower(), r.depth_top_m,
                                                   r.depth_bottom_m), []) if sid not in used]
        if not cands:
            continue
        rel_days = rel_by_sensor[(r.network, r.their_station, r.their_sensor)]
        shares = pd.Series({sid: equal_share(sid, rel_days) for sid in cands})
        best = shares.max()
        if best > 0 and (shares == best).sum() == 1:
            by_data.append({"our_sensor_id": shares.idxmax(), "network": r.network,
                            "their_station": r.their_station, "their_sensor": r.their_sensor, "match": "data"})
            used.add(shares.idxmax())
    pairs = pd.concat([pairs, pd.DataFrame(by_data)], ignore_index=True)
    covered = rel_sensors.merge(pairs, on=["network", "their_station", "their_sensor"])
    rows = [("Morgan soil moisture sensors (metadata.csv)", len(meta)),
            (f"our soil moisture sensors (unit = {cfg.prepare.sensor_unit})", len(ours)),
            ("paired by exact name", int((pairs["match"] == "name").sum())),
            ("paired by unique depth", int((pairs["match"] == "depth").sum())),
            ("paired by data (replicate probes)", int((pairs["match"] == "data").sum())),
            ("sensors with released events", len(rel_sensors)),
            ("of those, paired with ours", len(covered)),
            ("released events at paired sensors",
             int(released.merge(covered, left_on=["network", "station", "sensor"],
                                right_on=["network", "their_station", "their_sensor"]).shape[0])),
            ("released events", len(released))]

    # Texture: our USDA class and fractions vs Morgan's metadata for paired sensors.
    m = (pairs.merge(ours[["sensor_id", "texture_class", "sand_frac"]], left_on="our_sensor_id", right_on="sensor_id")
         .merge(meta[["network", "station", "sensor", "sand_fraction", "soil_texture"]]
                .rename(columns={"station": "their_station", "sensor": "their_sensor"}),
                on=["network", "their_station", "their_sensor"]))
    their_texture = m["soil_texture"].fillna("unknown").str.lower().replace({"not available": "unknown"})
    rows += [("paired: texture class equal", (m["texture_class"] == their_texture).mean()),
             ("paired: median |sand difference| [fraction]",
              (m["sand_frac"] - m["sand_fraction"] / ismn_percent).abs().median())]

    # Daily theta on released event days of paired sensors.
    cmp_days = long.merge(covered[["network", "their_station", "their_sensor", "our_sensor_id"]],
                          on=["network", "their_station", "their_sensor"])
    daily = pd.concat([our_daily(sid).assign(our_sensor_id=sid) for sid in cmp_days["our_sensor_id"].unique()],
                      ignore_index=True)
    cmp_days = cmp_days.merge(daily, on=["our_sensor_id", "date"], how="left")
    diff = (cmp_days["theta_m3m3"] - cmp_days["released"]).abs()
    same = np.isclose(cmp_days["theta_m3m3"], cmp_days["released"], rtol=0, atol=np.finfo(np.float32).eps)
    by_sensor = cmp_days.assign(same=same).groupby("our_sensor_id")["same"].mean()
    rows += [("released event-days compared", len(cmp_days)),
             ("event-days missing in ours", int(cmp_days["theta_m3m3"].isna().sum())),
             ("event-days equal (float32 precision)", same.mean()),
             ("median |difference| theta", diff.median()), ("mean |difference| theta", diff.mean()),
             ("sensors with all event-days equal", (by_sensor == 1).mean())]

    # Sensor constants from our daily theta vs the released per-event constants.
    day_tbl = daily.rename(columns={"our_sensor_id": "sensor_id"})
    consts = sensor_constants(ours[ours["sensor_id"].isin(day_tbl["sensor_id"].unique())], day_tbl, cfg.soil)
    rel_c = (released.merge(covered, left_on=["network", "station", "sensor"],
                            right_on=["network", "their_station", "their_sensor"])
             .groupby("our_sensor_id")[["min_sm", "max_sm", "theta_fc"]].first()
             .rename_axis("sensor_id").reset_index())
    cc = consts.merge(rel_c, on="sensor_id")
    for ours_col, theirs_col in [("min_sm_m3m3", "min_sm"), ("max_sm_m3m3", "max_sm"), ("theta_fc_m3m3", "theta_fc")]:
        rows += [(f"sensor constants: {theirs_col} equal (float32 precision)",
                  np.isclose(cc[ours_col], cc[theirs_col], rtol=0, atol=np.finfo(np.float32).eps).mean()),
                 (f"sensor constants: median |difference| {theirs_col}", (cc[ours_col] - cc[theirs_col]).abs().median())]
    summary = pd.DataFrame(rows, columns=["quantity", "value"])
    log.info("ISMN check:\n%s", summary.to_string(index=False))
    cmp_days.to_parquet(run_dir / "event_day_theta.parquet", index=False)
    cc.to_parquet(run_dir / "sensor_constants.parquet", index=False)
    pairs.to_parquet(run_dir / "sensor_pairs.parquet", index=False)
    report = write_report(run_dir / "report.md", "ISMN check: prepare step 1 vs Morgan et al. inputs", [
        ("Summary", markdown_table(summary, "{:.4g}")),
        ("Notes", "Sensors are paired by exact name, else by unique (network, station, depth) with "
                  "stations compared case-insensitively, else (replicate probes) by the candidate whose "
                  "daily theta equals the released event values on the most days. Morgan's records end "
                  "earlier than this archive, so min_sm and max_sm are expected to differ.")])
    log.info("report: %s", report)
    sys.exit(0)

# ── Tier 2: our steps 0-3 against the released events ─────────────────────────
if args.tier == 2:
    outputs = Path(cfg.paths.outputs_dir)
    manifest = json.loads((outputs / "runs" / args.run / "manifest.json").read_text())
    stage_keys = manifest["stage_keys"]
    events = read_stage(outputs / "cache", "events", stage_keys["events"])["events"]
    fits = read_stage(outputs / "cache", "fits", stage_keys["fits"])["fits"]
    q_fits = fits[fits["model"] == "q_stage2"][["event_id", "q", "k_per_day", "theta_star_m3m3"]]
    our_events = events.merge(q_fits, on="event_id", how="left")
    rel = pd.read_csv(data / "ismn_results_star_th0_04may.csv.zip",
                      usecols=["SITE_ID", "Sensor_ID", "event_start", "q_q", "q_k", "q_theta_star"])
    meta = pd.read_csv(data / "ISMN" / "metadata.csv", usecols=["station", "sensor", "network"])
    rel = rel.merge(meta, left_on=["SITE_ID", "Sensor_ID"], right_on=["station", "sensor"], validate="many_to_one")
    rel["sensor_id"] = rel["network"] + "/" + rel["SITE_ID"] + "/" + rel["Sensor_ID"]
    rel["start_date"] = pd.to_datetime(rel["event_start"])
    summary, matched = compare_events(our_events, rel, {"q": "q_q", "k_per_day": "q_k",
                                                        "theta_star_m3m3": "q_theta_star"})
    log.info("tier 2:\n%s", summary.to_string(index=False))
    matched.to_parquet(run_dir / "matched_events.parquet", index=False)
    report = write_report(run_dir / "report.md", f"Reproduction tier 2: run `{args.run}` vs released events", [
        ("Event and fit agreement", markdown_table(summary, "{:.4g}")),
        ("Notes", "Matching is on (network/station/sensor, start date). Sensor names come from the "
                  "ISMN archive, so renamed instruments in a newer archive do not match; the "
                  "'sensors in both' rows separate that from detection differences.")])
    log.info("report: %s", report)
    sys.exit(0)

# ── Released fits and the join recipe of Morgan's utils_read.add_info ─────────
# 1. Event theta arrays are stored as strings; parse them only to get the event
#    min and max (for frac_range). 2. Rename the id columns.
released = pd.read_csv(data / "ismn_results_star_th0_04may.csv.zip")
log.info("released table: %d events", len(released))
theta = released["theta"].map(lambda s: np.array(s.strip("[]").split(), dtype=float))
released["event_min_m3m3"] = theta.map(np.min)
released["event_max_m3m3"] = theta.map(np.max)
released = released.rename(columns={"SITE_ID": "station", "Sensor_ID": "sensor"})

# 3. Inner joins with the ISMN metadata (adds the network) and the station
#    ancillary table; 4. with the CCI land cover legend on LC = CCI_Code.
meta = pd.read_csv(data / "ISMN" / "metadata.csv", usecols=["station", "sensor", "network"])
anc = pd.read_csv(data / "ISMN" / "ismn_ancillary.csv", usecols=["network", "station"])
cci = pd.read_csv(data / "cci_landcover.csv", usecols=["CCI_Code"])
df = (released.merge(meta, on=["station", "sensor"], validate="many_to_one")
      .merge(anc, on=["network", "station"], validate="many_to_one")
      .merge(cci, left_on="LC", right_on="CCI_Code", validate="many_to_one"))
log.info("after metadata, ancillary and CCI joins: %d events", len(df))

# 5. Inner join with MCD12Q1 at the station pixel for the event start year.
lc = pd.read_csv(data / "ISMN" / "ancillary" / "LC" / "ismn_MCD12Q1_061_LC.csv",
                 usecols=["network", "station", "Date", "LC_IGBP"])
lc["year"] = pd.to_datetime(lc["Date"]).dt.year
df["start_date"] = pd.to_datetime(df["event_start"])
df["year"] = df["start_date"].dt.year
df = df.merge(lc[["network", "station", "year", "LC_IGBP"]], on=["network", "station", "year"],
              validate="many_to_one")
log.info("after MCD12Q1 join: %d events", len(df))
ours.append(("after_lc_join", "", "events", len(df)))

# 6. IGBP number -> IGBP code -> vegetation class (covariates.class_scheme).
igbp = pd.read_csv(TABLES_DIR / "igbp_classes.csv")
df["igbp_class"] = df["LC_IGBP"].map(igbp.set_index("igbp_number")["igbp_class"])
df["veg_class"] = assign_class(df["igbp_class"], cfg.covariates.class_scheme)

# ── Rename to our `analysis` columns (released q model = our q_stage2 fit) ─────
depth = df["depth"].str.strip("()").str.split(",", expand=True).astype(float)
analysis = pd.DataFrame({
    "event_id": df["network"] + "/" + df["station"] + "/" + df["sensor"] + "/" + df["event_start"],
    "sensor_id": df["network"] + "/" + df["station"] + "/" + df["sensor"],
    "station_id": df["network"] + "/" + df["station"],
    "network": df["network"], "lat_deg": df["LAT"], "lon_deg": df["LON"],
    "start_date": df["start_date"], "duration_days": df["duration"],
    "q": df["q_q"], "k_per_day": df["q_k"], "theta_star_m3m3": df["q_theta_star"],
    "theta_0_m3m3": df["q_theta_0"], "theta_w_m3m3": df["q_theta_w"], "r2": df["q_r_squared"],
    "min_sm_m3m3": df["min_sm"], "max_sm_m3m3": df["max_sm"], "theta_fc_m3m3": df["theta_fc"],
    "dz_m": df["dz"], "depth_top_m": depth[0], "depth_bottom_m": depth[1],
    "event_min_m3m3": df["event_min_m3m3"], "event_max_m3m3": df["event_max_m3m3"],
    "ai_60d": df["AI_60d"], "lai_m2m2": df["LAI"],
    "igbp_class": df["igbp_class"], "veg_class": df["veg_class"],
})
analysis = add_derived(analysis)
assert np.allclose(analysis["mean_depth_m"], df["z_m"]), "mean depth should equal released z_m"

# ── Steps 5-8 exactly as analyze.py runs them ─────────────────────────────────
analysis = flag_filters(analysis, cfg.filters)
steps = cascade(analysis)
log.info("filter cascade:\n%s", steps.to_string(index=False))
after_quality = steps.set_index("after").loc[QUALITY_RULES[-1]]
ours += [("after_quality_filters", "", "events", after_quality["events"]),
         ("after_quality_filters", "", "stations", after_quality["stations"])]

final = analysis[analysis["in_final_set"]]
ours += [("final_set", "", "events", len(final)),
         ("final_set", "", "stations", final["station_id"].nunique())]
for veg_class, q in final.groupby("veg_class")["q"].median().items():
    ours.append(("median_q", veg_class, "q", q))

in_surface = surface_set(analysis, cfg.filters.lai_min_m2m2, cfg.binning.lai_min_inclusive_surface)
surface_events = analysis[in_surface]
ours += [("surface_set", "", "events", len(surface_events)),
         ("surface_set", "", "stations", surface_events["station_id"].nunique())]
bins = bin_medians(surface_events, cfg.filters.classes, cfg.binning)
surfaces = fit_surface(bins, cfg.filters.classes, cfg.binning)
for row in surfaces.itertuples():
    ours += [("bins_used", row.veg_class, "n_bins", row.n_bins),
             ("events_in_bins", row.veg_class, "n_events", row.n_events)]
    ours += [("surface", row.veg_class, k, getattr(row, k)) for k in ["a", "b", "c", "d", "r2_bins"]]

analysis.to_parquet(run_dir / "analysis.parquet", index=False)
bins.to_parquet(run_dir / "bins.parquet", index=False)
surfaces.to_parquet(run_dir / "surfaces.parquet", index=False)

# ── Fit check: refit a random sample of released events with our fit_event ────
# Inputs are the released event arrays and constants (theta, t, theta_fc,
# min_sm, dz, q_theta_w, and the released scalar `pet` = event PET_max).
# Reported, not pass/fail: k and theta_star are not separately identifiable.
refit_rows = []
if args.refit_sample:
    sample = released.sample(min(args.refit_sample, len(released)), random_state=0)
    for r in sample.itertuples():
        t = np.array(r.time.strip("[]").split(), dtype=float)
        sensor = {"theta_fc_m3m3": r.theta_fc, "min_sm_m3m3": r.min_sm, "dz_m": r.dz,
                  "theta_w_m3m3": r.q_theta_w if np.isfinite(r.q_theta_w) else r.min_sm}
        theta_r = theta.loc[r.Index]
        for model, pairs in [("q_stage2", [("q", "q_q"), ("k_per_day", "q_k"),
                                           ("theta_star_m3m3", "q_theta_star"), ("r2", "q_r_squared")]),
                             ("exponential", [("tau_days", "exp_tau"), ("theta_w_m3m3", "exp_theta_w"),
                                              ("r2", "exp_r_squared")])]:
            fit = fit_event(t, theta_r, np.array([r.pet]), sensor, model, cfg.fitting)
            for ours_col, theirs_col in pairs:
                refit_rows.append({"model": model, "parameter": ours_col, "ours_ok": fit["fit_status"] == "ok",
                                   "ours": fit.get(ours_col, np.nan), "released": getattr(r, theirs_col)})
refit = pd.DataFrame(refit_rows)
refit_table = pd.DataFrame()
if not refit.empty:
    refit["abs_diff"] = (refit["ours"] - refit["released"]).abs()
    refit["both_ok"] = refit["ours_ok"] & refit["released"].notna()
    refit_table = refit.groupby(["model", "parameter"], sort=False).agg(
        events=("ours", "size"),
        ours_failed=("ours_ok", lambda ok: int((~ok).sum())),
        released_failed=("released", lambda v: int(v.isna().sum())),
        median_abs_diff=("abs_diff", "median"), mean_abs_diff=("abs_diff", "mean"),
        max_abs_diff=("abs_diff", "max")).reset_index()
    log.info("refit of %d released events (diffs over events where both fitted):\n%s",
             len(sample), refit_table.to_string(index=False))

# ── Compare with the published numbers ────────────────────────────────────────
targets = pd.read_csv(TABLES_DIR / "morgan2025_tier1_targets.csv", keep_default_na=False)
ours = pd.DataFrame(ours, columns=["check", "veg_class", "quantity", "ours"])
cmp = targets.merge(ours, on=["check", "veg_class", "quantity"], how="left")
cmp["expected"] = cmp["expected"].astype(float)
cmp["tolerance"] = cmp["tolerance"].astype(float)
cmp["ours"] = cmp["ours"].astype(float)
cmp["diff"] = cmp["ours"] - cmp["expected"]
cmp["pass"] = cmp["diff"].abs() <= cmp["tolerance"]  # tolerance is inclusive
cmp["pass"] = cmp["pass"].map({True: "yes", False: "NO"})
table = cmp[["check", "veg_class", "quantity", "expected", "ours", "diff", "tolerance", "pass"]]
log.info("tier 1 comparison:\n%s", table.to_string(index=False))

report = write_report(run_dir / "report.md", "Reproduction tier 1: downstream steps on the released fits", [
    ("Comparison with Morgan et al. (2025)", markdown_table(table, "{:.4g}")),
    ("Filter cascade", markdown_table(steps)),
    (f"Fit check: {args.refit_sample} released events refitted with our fit_event",
     markdown_table(refit_table, "{:.2e}") if not refit_table.empty else "skipped"),
    ("Surfaces", markdown_table(surfaces)),
])
log.info("report: %s", report)
sys.exit(0 if (cmp["pass"] == "yes").all() else 1)
