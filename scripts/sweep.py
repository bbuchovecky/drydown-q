"""Sensitivity sweep: many analyze.py runs -> summary table + specification curve.

    pixi run sweep --sweep thresholds
    python scripts/sweep.py --sweep <name> [--paths config/paths.yml]

The sweep file (config/sweeps/<name>.yml) names a base run and the settings to
vary (one at a time, or a grid). Every configuration is an ``analyze.py`` run
in ``outputs/runs/sweep_<sweep>/<label>/``; all share ``outputs/cache``, so a
configuration that changes only filters or binning reuses the base run's
events and fits. The base configuration runs first, alone, with the full
``runtime.n_jobs`` (it fills the shared stages); the rest then run in parallel
with joblib, each single-threaded (one level of parallelism). A configuration
already completed with the same code is skipped, and a failed one is recorded
in the summary and does not stop the sweep.

Outputs in ``outputs/runs/sweep_<sweep>/``: ``summary.parquet`` (one row per
configuration) and ``specification_curve.png``.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from drydown_q.config import REPO_ROOT, code_hash, load_config, load_sweep
from drydown_q.schemas import assert_schema

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--sweep", required=True, help="sweep name: config/sweeps/<sweep>.yml")
parser.add_argument("--paths", default=None, help="paths file (default config/paths.yml)")
args = parser.parse_args()

base_run, configs = load_sweep(args.sweep)
cfg = load_config(base_run, args.paths)
sweep_dir = Path(cfg.paths.outputs_dir) / "runs" / f"sweep_{args.sweep}"
sweep_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=cfg.runtime.log_level, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    force=True, handlers=[logging.StreamHandler(),
                                          logging.FileHandler(sweep_dir / "sweep.log", mode="a")])
log = logging.getLogger("sweep")
code = code_hash()
classes = cfg.filters.classes
log.info("sweep %s: base run %s, %d configurations", args.sweep, base_run, len(configs))


def run_one(label: str, overrides: dict, n_jobs: int) -> dict:
    """Run analyze.py for one configuration (skipped if already done with this code)."""
    name = f"sweep_{args.sweep}/{label}"
    summary_file = Path(cfg.paths.outputs_dir) / "runs" / name / "summary.json"
    if summary_file.exists():
        done = json.loads(summary_file.read_text())
        if done.get("code_hash") == code and done.get("overrides", {}) == overrides_with_jobs(overrides, n_jobs):
            return {"label": label, "status": "ok", "message": "reused"}
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "analyze.py"), "--run", base_run, "--name", name,
           "--overrides", json.dumps(overrides_with_jobs(overrides, n_jobs))]
    if args.paths:
        cmd += ["--paths", args.paths]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        return {"label": label, "status": "failed", "message": result.stderr.strip().splitlines()[-1]}
    return {"label": label, "status": "ok", "message": ""}


def overrides_with_jobs(overrides: dict, n_jobs: int) -> dict:
    return {**overrides, "runtime": {**overrides.get("runtime", {}), "n_jobs": n_jobs}}


# Base configuration first (fills the shared cache), then the rest in parallel.
outcomes = [run_one(*configs[0], cfg.runtime.n_jobs)]
log.info("  %s: %s %s", configs[0][0], outcomes[0]["status"], outcomes[0]["message"])
outcomes += Parallel(n_jobs=cfg.runtime.n_jobs)(delayed(run_one)(label, ov, 1) for label, ov in configs[1:])
for o in outcomes[1:]:
    log.info("  %s: %s %s", o["label"], o["status"], o["message"])

# ── Summary table ─────────────────────────────────────────────────────────────
rows = []
for (label, overrides), outcome in zip(configs, outcomes):
    changed = {f"{s}.{k}": v for s, kv in overrides.items() for k, v in kv.items()}
    row = {"run": label, "changed_key": ", ".join(changed) or "(baseline)",
           "changed_value": ", ".join(json.dumps(v) for v in changed.values()),
           "status": outcome["status"], "message": outcome["message"],
           "events_final": np.nan, "stations_final": np.nan, "events_surface": np.nan}
    summary_file = sweep_dir / label / "summary.json"
    if outcome["status"] == "ok" and summary_file.exists():
        s = json.loads(summary_file.read_text())
        row.update(events_final=s["events_final"], stations_final=s["stations_final"],
                   events_surface=s["events_surface"])
        for c in classes:
            row[f"median_q_{c}"] = s["median_q"].get(c, np.nan)
            surf = s["surfaces"].get(c, {})
            for k in ["a", "b", "c", "d", "r2_bins", "n_bins", "dq_dai_at_median_lai"]:
                row[f"{k}_{c}"] = surf.get(k, np.nan)
            row[f"dq_dai_sign_{c}"] = np.sign(row[f"dq_dai_at_median_lai_{c}"])
    rows.append(row)
summary = pd.DataFrame(rows)
# Headline contrast: grassland minus woodland median q (the paper's HERB vs WOOD).
if {"median_q_HERB", "median_q_WOOD"} <= set(summary.columns):
    summary["herb_minus_wood_q"] = summary["median_q_HERB"] - summary["median_q_WOOD"]
for column in summary.columns:
    if column not in ("run", "changed_key", "changed_value", "status", "message"):
        summary[column] = summary[column].astype(float)
assert_schema(summary, "summary")
summary.to_parquet(sweep_dir / "summary.parquet", index=False)
log.info("summary: %s (%d ok, %d failed)", sweep_dir / "summary.parquet",
         (summary.status == "ok").sum(), (summary.status != "ok").sum())

# ── Specification curve ───────────────────────────────────────────────────────
# Runs are ranked by the HERB - WOOD difference in median q (all panels share
# that order); the dot matrix below marks which setting each run changes.
ok = summary[summary.status == "ok"].copy()
order = "herb_minus_wood_q" if "herb_minus_wood_q" in ok else f"median_q_{classes[0]}"
ok = ok.sort_values(order, ignore_index=True)
x = np.arange(len(ok))
panels = [("median q", [f"median_q_{c}" for c in classes]), ("HERB − WOOD q", ["herb_minus_wood_q"]),
          *[(f"surface {k}", [f"{k}_{c}" for c in classes]) for k in ["a", "b", "c", "d"]],
          ("R² on bins", [f"r2_bins_{c}" for c in classes]),
          ("sign ∂q/∂AI at median LAI", [f"dq_dai_sign_{c}" for c in classes])]
settings = [k for k in dict.fromkeys(ok["changed_key"]) if k != "(baseline)"]
fig, axes = plt.subplots(len(panels) + 1, 1, sharex=True, figsize=(max(6, 0.35 * len(ok) + 2), 2 * len(panels) + 3),
                         gridspec_kw={"height_ratios": [1] * len(panels) + [max(1, 0.25 * len(settings))]})
base_x = x[ok["run"] == "baseline"]
colours = dict(zip(classes, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
for ax, (title, columns) in zip(axes, panels):
    for column in columns:
        if column in ok:
            veg_class = column.rsplit("_", 1)[-1]
            ax.plot(x, ok[column], "o", ms=4, color=colours.get(veg_class, "k"), label=veg_class)
    for bx in base_x:
        ax.axvline(bx, color="0.6", lw=0.8, ls="--")
    ax.set_ylabel(title, fontsize=8)
axes[0].legend(fontsize=7, ncol=len(classes), loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False)
matrix = axes[-1]
for i, row in ok.iterrows():
    if row["changed_key"] in settings:
        matrix.plot(i, settings.index(row["changed_key"]), "ks", ms=5)
        matrix.annotate(row["changed_value"], (i, settings.index(row["changed_key"])), fontsize=6,
                        xytext=(0, 5), textcoords="offset points", ha="center")
matrix.set_yticks(range(len(settings)), settings, fontsize=7)
matrix.set_ylim(-1, len(settings))
matrix.set_xticks(x)
matrix.set_xlabel(f"run rank (by {order}); dashed line = baseline", fontsize=8)
fig.suptitle(f"Specification curve: sweep {args.sweep} (base run {base_run})", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(sweep_dir / "specification_curve.png", dpi=150)
log.info("figure: %s", sweep_dir / "specification_curve.png")
