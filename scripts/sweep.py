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

from joblib import Parallel, delayed

from drydown_q.config import REPO_ROOT, code_hash, load_config, load_sweep
from drydown_q.report import specification_curve, sweep_row, sweep_summary

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

# ── Summary table and specification curve ─────────────────────────────────────
rows = []
for (label, overrides), outcome in zip(configs, outcomes):
    summary_file = sweep_dir / label / "summary.json"
    result = json.loads(summary_file.read_text()) if outcome["status"] == "ok" and summary_file.exists() else None
    rows.append(sweep_row(label, overrides, outcome["status"], outcome["message"], result, classes))
summary = sweep_summary(rows)
summary.to_parquet(sweep_dir / "summary.parquet", index=False)
log.info("summary: %s (%d ok, %d failed)", sweep_dir / "summary.parquet",
         (summary.status == "ok").sum(), (summary.status != "ok").sum())
figure = specification_curve(summary, classes, sweep_dir / "specification_curve.png",
                             f"Specification curve: sweep {args.sweep} (base run {base_run})")
log.info("figure: %s", figure)
