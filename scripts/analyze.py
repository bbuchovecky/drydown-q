"""Drydown analysis of Morgan et al. (2025), from station_day to q surfaces.

This file is the methods section. It defines no functions: each numbered block
states one decision (logged with the values from the resolved config), calls
one or two library functions, and logs what remains.

    pixi run analyze --run morgan2025
    python scripts/analyze.py --run <name> [--paths config/paths.yml]

Every step goes through ``cached_stage``: its outputs are reused when the
stage's config section, the upstream stage and the code under src/ are
unchanged. Outputs: ``<outputs_dir>/cache/<stage>/<key>/`` and
``<outputs_dir>/runs/<run>/`` (report, resolved config, manifest, log).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from drydown_q.binning import bin_medians, fit_surface, surface_set
from drydown_q.cache import cached_stage
from drydown_q.config import (as_dict, code_hash, config_hash, differences_from_default,
                              load_config, write_resolved)
from drydown_q.covariates import attach_covariates
from drydown_q.events import find_events
from drydown_q.filters import RULES, cascade, flag_filters
from drydown_q.fitting import fit_all
from drydown_q.report import file_checksum, markdown_table, write_manifest, write_report
from drydown_q.schemas import assert_schema
from drydown_q.soil import sensor_constants

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--run", required=True, help="run name: config/runs/<run>.yml")
parser.add_argument("--paths", default=None, help="paths file (default config/paths.yml)")
parser.add_argument("--overrides", default=None, help="JSON overrides (used by sweep.py)")
parser.add_argument("--name", default=None, help="run folder under outputs/runs/ (default: --run)")
args = parser.parse_args()
started = time.time()

# ── Step 0. Configuration and inputs ───────────────────────────────────────────
# Settings are default.yml overridden by config/runs/<run>.yml; every setting that
# differs from the Morgan baseline is printed. Inputs are the `sensors` and
# `station_day` tables written by scripts/prepare.py (paths.prepared_dir).
cfg = load_config(args.run, args.paths, json.loads(args.overrides) if args.overrides else None)
outputs = Path(cfg.paths.outputs_dir)
run_dir, cache_dir = outputs / "runs" / (args.name or cfg.run_name), outputs / "cache"
run_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=cfg.runtime.log_level, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    force=True, handlers=[logging.StreamHandler(),
                                          logging.FileHandler(run_dir / "analyze.log", mode="w")])
log = logging.getLogger("analyze")
write_resolved(cfg, run_dir / "config.resolved.yml")
diffs = differences_from_default(cfg)
log.info("run %s: %d setting(s) differ from the Morgan baseline", cfg.run_name, len(diffs))
for key, (default, value) in diffs.items():
    log.info("    %s = %r (default %r)", key, value, default)

prepared = Path(cfg.paths.prepared_dir)
input_files = {name: prepared / f"{name}.parquet" for name in ("sensors", "station_day")}
inputs_sha = {name: file_checksum(path) for name, path in input_files.items()}
sensors = pd.read_parquet(input_files["sensors"])
station_day = pd.read_parquet(input_files["station_day"])
assert_schema(sensors, "sensors")
assert_schema(station_day, "station_day")
code = code_hash()
keys = {"inputs": config_hash(inputs_sha)}
report = []
log.info("step 0: %d sensors at %d stations, %d sensor-days",
         len(sensors), sensors["station_id"].nunique(), len(station_day))

# ── Step 1. Sensor constants ───────────────────────────────────────────────────
# theta_fc from the Clapp & Hornberger (1978) texture class at psi_fc (max_sm if
# the class has no parameters); max_sm = a high quantile of daily theta; theta_w
# by `soil.theta_w_rule`; dz by `soil.dz_rule`.
s = cfg.soil
log.info("step 1: theta_fc = %s at %g kPa (rounded to %g), max_sm = q%g, theta_w = %s, dz = %s",
         s.theta_fc_rule, s.psi_fc_kpa, s.theta_fc_round_m3m3, s.max_sm_quantile, s.theta_w_rule, s.dz_rule)
tables, keys["sensor_constants"] = cached_stage(
    "sensor_constants", lambda: {"sensor_constants": sensor_constants(sensors, station_day, s)},
    cache_dir, s, keys["inputs"], code)
constants = tables["sensor_constants"]
fallback = (constants["theta_fc_source"] == "max_sm").mean()
log.info("        %d sensors; %.1f%% use the max_sm fallback for theta_fc", len(constants), 100 * fallback)

# ── Step 2. Detect drydown events ──────────────────────────────────────────────
e = cfg.events
log.info("step 2: start = %d consecutive days with dtheta < -%g mm/d / (1000 dz) and P <= %g mm "
         "(rain checked in the %s); end at the first day with dtheta > -%g mm/d / (1000 dz); "
         "theta > theta_fc masked: %s", e.min_decline_days, e.start_min_decline_mm_day,
         e.precip_max_mm_day, e.precip_check, e.end_min_decline_mm_day, e.mask_above_fc)
tables, keys["events"] = cached_stage(
    "events", lambda: find_events(station_day, constants, e), cache_dir, e, keys["sensor_constants"], code)
events, event_days = tables["events"], tables["event_days"]
log.info("        precipitation source by sensor: %s", tables["precip_sources"]["source"].value_counts().to_dict())
reasons = events["end_reason"].value_counts().to_dict()
log.info("        %d events at %d sensors; end reasons %s; mean days above theta_fc %.2f",
         len(events), events["sensor_id"].nunique(), reasons, events["n_days_above_fc"].mean())

# ── Step 3. Fit the loss models ────────────────────────────────────────────────
f = cfg.fitting
log.info("step 3: models %s by bounded least squares in theta; q model free %s, theta_0 = min(event max, "
         "theta_fc), theta_w fixed; q in %s from %g; theta_star in %s; k in [0, PET_max/(1000 dz)]",
         f.models, f.free_params_q, f.q_bounds, f.q_start, f.theta_star_bounds)
tables, keys["fits"] = cached_stage(
    "fits", lambda: {"fits": fit_all(events, event_days, constants, f, cfg.runtime.n_jobs)},
    cache_dir, f, keys["events"], code)
fits = tables["fits"]
status = fits.groupby("model")["fit_status"].value_counts().unstack(fill_value=0)
at_bound = fits.groupby("model")[[c for c in fits.columns if c.startswith("at_bound_")]].mean()
log.info("        fit status by model:\n%s", status.to_string())
log.info("        share at a bound by model:\n%s", at_bound.to_string(float_format="{:.3f}".format))
report.append(("Step 3: fits", "Fit status:\n\n" + markdown_table(status.reset_index()) +
               "\n\nShare of fits with each parameter at a bound:\n\n" +
               markdown_table(at_bound.reset_index())))

# ── Step 4. Attach covariates and vegetation classes ──────────────────────────
c = cfg.covariates
log.info("step 4: analysed model %s; AI = sum(P)/sum(PET) over the %d days ending %s the start day (%d valid days needed); "
         "LAI = %s; class = %s of the IGBP class in the %s", c.analysis_model, c.ai_window_days,
         "on" if c.ai_window_includes_start else "before", c.ai_window_min_valid_days,
         c.lai_statistic, c.class_scheme, c.landcover_year.replace("_", " "))
# The analysed model's fits (covariates.analysis_model) are the ones filtered and binned.
fits_q = fits[fits["model"] == c.analysis_model].drop(columns="model")
tables, keys["covariates"] = cached_stage(
    "covariates",
    lambda: {"covariates": attach_covariates(fits_q, events, event_days, sensors, constants, station_day, c)},
    cache_dir, c, keys["fits"], code)
covs = tables["covariates"]
missing = covs[["ai_60d", "lai_m2m2", "veg_class"]].isna().mean()
log.info("        %d events; missing AI %.1f%%, LAI %.1f%%, class %.1f%%", len(covs),
         100 * missing["ai_60d"], 100 * missing["lai_m2m2"], 100 * missing["veg_class"])

# ── Step 5. Flag the filters ──────────────────────────────────────────────────
fl = cfg.filters
log.info("step 5: R2 >= %g, duration in %s d, q > %g, frac_range > %g, mean depth <= %g m, dz <= %g m, "
         "theta_star - min_sm >= %g, theta_star - theta_w >= %g, |lat| <= %s, class in %s, LAI %s %g",
         fl.r2_min, fl.duration_days, fl.q_min, fl.frac_range_min, fl.mean_depth_max_m, fl.dz_max_m,
         fl.theta_star_minus_min_sm_min, fl.theta_star_minus_w_min, fl.abs_lat_max_deg, fl.classes,
         ">=" if fl.lai_min_inclusive else ">", fl.lai_min_m2m2)
tables, keys["analysis"] = cached_stage(
    "analysis", lambda: {"analysis": flag_filters(covs, fl)}, cache_dir, fl, keys["covariates"], code)
analysis = tables["analysis"]
steps = cascade(analysis)
log.info("        cascade (events and stations after each rule, in order):\n%s", steps.to_string(index=False))
report.append(("Step 5: filter cascade", markdown_table(steps)))

# ── Step 6. Final event set ────────────────────────────────────────────────────
final = analysis[analysis["in_final_set"]]
medians = final.groupby("veg_class")["q"].agg(["size", "median"]).rename(
    columns={"size": "events", "median": "median_q"}).reset_index()
log.info("step 6: final set = all %d rules: %d events at %d stations; median q by class:\n%s",
         len(RULES), len(final), final["station_id"].nunique(), medians.to_string(index=False))
report.append(("Step 6: final event set",
               f"{len(final)} events at {final['station_id'].nunique()} stations.\n\n" + markdown_table(medians)))

# ── Step 7. Bin by AI x LAI within classes ────────────────────────────────────
b = cfg.binning
in_surface = surface_set(analysis, fl.lai_min_m2m2, b.lai_min_inclusive_surface)
log.info("step 7: surface set = final set with LAI %s %g (%d events, %d stations); AI edges %s, LAI edges %s, "
         "%s-closed; median q per bin; bins with >= %d events are used",
         ">=" if b.lai_min_inclusive_surface else ">", fl.lai_min_m2m2, in_surface.sum(),
         analysis.loc[in_surface, "station_id"].nunique(), b.ai_edges, b.lai_edges,
         "right" if b.bins_right_closed else "left", b.min_events_per_bin)
tables, keys["bins"] = cached_stage(
    "bins", lambda: {"bins": bin_medians(analysis[in_surface], fl.classes, b)},
    cache_dir, b, keys["analysis"], code)
bins = tables["bins"]
used = bins[bins["used_in_fit"]].groupby("veg_class").agg(bins_used=("n_events", "size"),
                                                           events_in_bins=("n_events", "sum")).reset_index()
log.info("        bins used per class:\n%s", used.to_string(index=False))

# ── Step 8. Fit the q surfaces ─────────────────────────────────────────────────
log.info("step 8: q = %s fitted to the bin medians by least squares, %s bin weights", b.surface_form, b.bin_weights)
tables, keys["surfaces"] = cached_stage(
    "surfaces", lambda: {"surfaces": fit_surface(bins, fl.classes, b)}, cache_dir, b, keys["bins"], code)
surfaces = tables["surfaces"]
# Headline: the sign of dq/dAI = a + c * LAI at each class's median LAI (surface set).
lai_median = analysis[in_surface].groupby("veg_class")["lai_m2m2"].median()
surfaces["lai_median_m2m2"] = surfaces["veg_class"].map(lai_median)
surfaces["dq_dai_at_median_lai"] = surfaces["a"] + surfaces["c"] * surfaces["lai_median_m2m2"]
for row in surfaces.itertuples():
    log.info("        %-5s a=%.3f±%.3f b=%.3f±%.3f c=%.3f±%.3f d=%.3f±%.3f R2(bins)=%.2f n_bins=%d [%s]",
             row.veg_class, row.a, row.a_se, row.b, row.b_se, row.c, row.c_se, row.d, row.d_se,
             row.r2_bins, row.n_bins, row.fit_status)
report.append(("Step 7: bins used", markdown_table(used)))
report.append(("Step 8: surfaces", markdown_table(surfaces)))

# ── Step 9. Report ─────────────────────────────────────────────────────────────
summary = {"run": args.name or cfg.run_name, "base_run": cfg.run_name, "code_hash": code,
           "overrides": json.loads(args.overrides) if args.overrides else {},
           "changed": {k: v[1] for k, v in diffs.items()},
           "events_detected": len(events), "events_final": len(final),
           "stations_final": int(final["station_id"].nunique()),
           "events_surface": int(in_surface.sum()),
           "median_q": dict(zip(medians["veg_class"], medians["median_q"])),
           "surfaces": surfaces.drop(columns="fit_status").set_index("veg_class").to_dict("index"),
           "stage_keys": keys}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
path = write_report(run_dir / "report.md", f"drydown-q run `{cfg.run_name}`", [
    ("Settings that differ from the Morgan baseline",
     "\n".join(f"- `{k}` = {v[1]!r} (default {v[0]!r})" for k, v in diffs.items()) or "none"),
    ("Step 1: sensor constants", f"{len(constants)} sensors; {100 * fallback:.1f}% use the max_sm fallback."),
    ("Step 2: events", f"{len(events)} events; end reasons {reasons}."),
    *report])
manifest = write_manifest(run_dir, cfg, code, inputs_sha, keys, started)
log.info("step 9: report %s (manifest %s)", path, manifest)
