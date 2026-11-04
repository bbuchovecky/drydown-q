"""Drydown analysis of Morgan et al. (2025), from station_day to q surfaces.

This file is the methods section. It defines no functions: each numbered block
states one decision (with the values taken from the resolved config), calls
one or two library functions, and logs what remains.

    pixi run analyze --run morgan2025
    python scripts/analyze.py --run <name> [--paths config/paths.yml]

Every step after step 0 goes through ``cached_stage``: its outputs are reused
when the stage's config section, the upstream stage and the code under src/
are unchanged. Outputs: ``<outputs_dir>/cache/<stage>/<key>/`` and
``<outputs_dir>/runs/<run>/`` (report, resolved config, manifest, log).
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from drydown_q.config import code_hash, config_hash, differences_from_default, load_config, write_resolved
from drydown_q.report import file_checksum, write_manifest
from drydown_q.schemas import assert_schema

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--run", required=True, help="run name: config/runs/<run>.yml")
parser.add_argument("--paths", default=None, help="paths file (default config/paths.yml)")
args = parser.parse_args()
started = time.time()

# ── Step 0. Configuration and inputs ───────────────────────────────────────────
# Settings are default.yml overridden by config/runs/<run>.yml; every setting that
# differs from the Morgan baseline is printed. Inputs are the `sensors` and
# `station_day` tables written by scripts/prepare.py (paths.prepared_dir).
cfg = load_config(args.run, args.paths)
outputs = Path(cfg.paths.outputs_dir)
run_dir, cache_dir = outputs / "runs" / cfg.run_name, outputs / "cache"
run_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=cfg.runtime.log_level, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S", force=True,
                    handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "analyze.log", mode="w")])
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
stage_keys = {"inputs": config_hash(inputs_sha)}
log.info("step 0: %d sensors at %d stations, %d sensor-days",
         len(sensors), sensors["station_id"].nunique(), len(station_day))

# Steps 1-9 (sensor constants, events, fits, covariates, filters, final set,
# bins, surfaces, report) are added in build phases 2-5.

manifest = write_manifest(run_dir, cfg, code, inputs_sha, stage_keys, started)
log.info("manifest: %s", manifest)
