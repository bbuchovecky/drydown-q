# Conventions for anyone (human or AI) changing this repository

This is a scientific analysis, not a software product. A scientist must be able
to read the code and see every assumption and decision, from loading data to
binning. Prefer readability over abstraction everywhere.

## Structure

- Plain functions that take and return pandas tables, with explicit parameters.
- No class hierarchies, agents, handlers, lazy properties, hidden caches,
  metaprogramming or plugin systems. Frozen dataclasses for config sections are
  the only classes.
- About a dozen modules in `src/drydown_q/`, each under ~300 lines, flat.
- Dependencies: numpy, pandas, scipy, xarray, netCDF4, pyarrow, matplotlib,
  pyyaml, joblib, pytest, ismn (pyproj from conda-forge only because ismn needs
  it). No Snakemake, no database, no CLI framework beyond argparse.
- `scripts/analyze.py` defines no functions; it reads as the methods section.

## Configuration

- Every scientific setting is in `config/default.yml`, one section per stage, with
  a one-line comment and a unit. The defaults are the Morgan et al. baseline as
  run; never change a default. Alternatives go in `config/runs/<name>.yml`.
- Machine-specific paths go in `config/paths.yml` (git-ignored).
- `config.py` only loads, merges and validates.
- No number with scientific meaning in a `.py` file
  (`tests/test_no_scientific_numbers.py` enforces this for `src/` and `scripts/`).
- Never hide a scientific choice in a function's default argument.
- Do not invent scientific values: if a decision needs one that is not in the
  config, ask.

## Data

- Units in column names: `theta_m3m3`, `precip_mm`, `pet_mm`, `dz_m`, `k_per_day`,
  `et_max_mm_day`, `lai_m2m2`. Convert units once, at load time.
- Each stage writes Parquet tables with fixed schemas (`schemas.py`, README).
- Arrays are stored as long tables (`event_days`), never as strings in cells.
- Flag, don't drop: each filter adds `pass_<rule>`; `in_final_set` is the AND of
  the active rules.

## Failures and logging

- Failures of single stations or events are recorded, never fatal.
- Every fit records `fit_status` and `fit_message`, and `at_bound_<param>` flags.
- Never use a bare `except`; catch the specific exception and record it.

## Code style

- Every model function's docstring states its equation, each variable and unit.
- Comments state decisions, not mechanics: which scientific choice a block makes
  and which config keys it reads.
- One level of parallelism: joblib over stations; a plain loop inside a station.
- Stage outputs go through `cache.cached_stage`; the key is the stage's config
  section + upstream key + hash of `src/`.

## Checks

- `pixi run smoke` must pass (< 1 minute) before any commit.
- Reproduction tier 1 coefficients must match the paper within 0.001; any mismatch
  is a bug in this code until shown otherwise.
