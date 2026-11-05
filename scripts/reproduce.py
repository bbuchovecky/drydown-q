"""Compare our pipeline with the released results of Morgan et al. (2025).

Tier 1 (downstream only): the released table
``<morgan_repo>/data/ismn_results_star_th0_04may.csv.zip`` stands in for our
``fits`` table, and analysis steps 4-9 (covariates, filters, final set, bins,
surfaces) run on it with the baseline config. Every number is compared with
``data/tables/morgan2025_tier1_targets.csv``; a mismatch is a bug in our code
until shown otherwise. Exit status 1 if any check fails.

Tier 2 (full rebuild from raw ISMN data) is added in build phase 5.

    pixi run reproduce
    python scripts/reproduce.py [--paths config/paths.yml]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from drydown_q.binning import bin_medians, fit_surface, surface_set
from drydown_q.config import TABLES_DIR, load_config, write_resolved
from drydown_q.covariates import add_derived, assign_class
from drydown_q.filters import QUALITY_RULES, cascade, flag_filters
from drydown_q.fitting import fit_event
from drydown_q.report import markdown_table, write_report

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--paths", default=None, help="paths file (default config/paths.yml)")
parser.add_argument("--refit-sample", type=int, default=0,
                    help="released events refitted with our fit_event (0 = skip)")
args = parser.parse_args()

cfg = load_config("morgan2025", args.paths)
run_dir = Path(cfg.paths.outputs_dir) / "runs" / "reproduce_tier1"
run_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=cfg.runtime.log_level, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S", force=True,
                    handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "reproduce.log", mode="w")])
log = logging.getLogger("reproduce")
write_resolved(cfg, run_dir / "config.resolved.yml")
data = Path(cfg.paths.morgan_repo) / "data"
ours = []   # (check, veg_class, quantity, value)

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
