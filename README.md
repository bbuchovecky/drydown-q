# drydown-q

A readable reimplementation of the soil moisture drydown analysis of
Morgan et al. (2025), *Nature Ecology & Evolution*,
[doi:10.1038/s41559-025-02810-8](https://doi.org/10.1038/s41559-025-02810-8).

It is built from Morgan et al.'s own code:
[`ecohydro/drydowns`](https://github.com/ecohydro/drydowns) (drydown detection and
model fitting) and [`ecohydro/ismn-drydowns`](https://github.com/ecohydro/ismn-drydowns)
(the ISMN analysis, released results and figures).

The pipeline:

1. Detect drydowns in ISMN in situ soil moisture.
2. Fit a nonlinear loss function to each event, with q as the shape exponent.
3. Attach covariates (antecedent aridity AI₆₀ and LAI).
4. Filter the events.
5. Bin by AI₆₀ × LAI within vegetation classes.
6. Fit q = a·AI₆₀ + b·LAI + c·AI₆₀·LAI + d to the bin medians.

`scripts/analyze.py` reads as the methods section. Every scientific setting is in
[`config/default.yml`](config/default.yml), and its defaults are the settings
Morgan et al. actually ran (inferred from the released results table).

## Quick start

```bash
pixi run smoke                     # tests + whole pipeline on a synthetic 3-station dataset
pixi run analyze --run morgan2025  # the baseline analysis (needs config/paths.yml and prepared data)
pixi run sweep --sweep thresholds  # sensitivity sweep, unattended
```

Before the first real run, copy `config/paths.example.yml` to `config/paths.yml`,
edit the paths, and run `pixi run prepare` once (slow: reads the raw archives).

| Task | Does |
| --- | --- |
| `smoke` | unit tests plus the whole pipeline on `tests/data/smoke` (< 1 min, no downloads) |
| `prepare` | raw ISMN, CHIRPS, dPET and MODIS data → `sensors` + `station_day` |
| `analyze --run <name>` | `station_day` → events, fits, filters, bins, surfaces, report |
| `reproduce` | baseline vs the released Morgan et al. results |
| `sweep --sweep <name>` | many `analyze` runs → `summary.parquet` + specification curve |
| `test` | `pytest` (synthetic recovery test is marked `recovery`) |
| `make-smoke-data` | regenerate the committed synthetic dataset (deterministic) |

## Inputs for `prepare`

| Path key | Expected contents |
| --- | --- |
| `ismn_archive` | an unpacked ISMN archive (header_values `.stm` files + `*_static_variables.csv`), read with the `ismn` package; its metadata cache goes to `<prepared_dir>/work/ismn_meta` |
| `chirps_dir` | CHIRPS v2.0 0.05° daily yearly files `chirps-v2.0.<year>.days_p05.nc`, variable `precip` [mm d⁻¹] |
| `dpet_dir` | dPET 0.1° daily yearly files `<year>_daily_pet.nc`, variable `pet` [mm d⁻¹] |
| `appeears_dir` | tidy AppEEARS parquet from `appeears_pipeline tidy`: `*__MOD15A2H.061.parquet` (or `MCD15A2H.061`), `*__MCD12Q1.061.parquet`, `*__MOD44B.061.parquet` |

`prepare` writes intermediate files to `<prepared_dir>/work/` (per-sensor daily means,
per-input-file extractions) and skips any that exist, so an interrupted run resumes.
Sensors or stations that fail are listed in `<prepared_dir>/failures.csv`.
`station_day` starts `ai_window_days − 1` days before each sensor's first value, so the
analysis can recompute the AI window sums exactly.

## Configuration

- `config/default.yml`: every scientific setting, one section per stage, with a unit
  and a comment on every key. Do not edit to explore; write a run file.
- `config/runs/<name>.yml`: overrides only the keys it lists. `morgan2025.yml` is empty.
- `config/paths.yml` (git-ignored): machine-specific paths; see `paths.example.yml`.
- `config/sweeps/<name>.yml`: sweep definitions.

The merge order is default → run → paths. Unknown keys and wrong types are rejected.
The resolved config is written to `outputs/runs/<run>/config.resolved.yml`.

## Outputs and caching

Each stage writes Parquet tables to `outputs/cache/<stage>/<key>/`, where the key
hashes the stage's config section, the upstream stage's key and the contents of
`src/`. A stage is recomputed only when one of those changes, so changing a filter
setting refits nothing. A run's report, resolved config, log and manifest (config,
code hash, input checksums, stage keys, package versions, wall time) go to
`outputs/runs/<run>/`.

## Sensitivity sweeps

`pixi run sweep --sweep thresholds` reads `config/sweeps/thresholds.yml` (a base run and
the settings to vary, one at a time or as a grid), runs every configuration through
`analyze.py` in `outputs/runs/sweep_<name>/<label>/`, and writes
`summary.parquet` and `specification_curve.png` there. The base configuration runs
first and fills the shared cache; the rest run in parallel, so a configuration that
changes only filters or binning reuses the base events and fits. A completed
configuration is skipped on rerun; a failed one is recorded and the sweep continues.

`python scripts/reproduce.py --sweep thresholds` runs the same sweep's filter and binning
configurations on the released Morgan et al. fits (tier 1 inputs, steps 5–8) and writes
`outputs/runs/released_sweep_thresholds/`; configurations that change event detection
are recorded as skipped.

## Reproduction

`pixi run reproduce` runs tier 1: the released Morgan et al. results table is used
as the `fits` table, steps 4–9 run on it with the baseline config, and every count,
median and coefficient is compared with `data/tables/morgan2025_tier1_targets.csv`.
The comparison is written to `outputs/runs/reproduce_tier1/report.md`; the script
exits non-zero if any check fails.

`python scripts/reproduce.py --check-ismn` compares the ISMN part of `prepare` (sensors,
daily θ, sensor constants) with Morgan's `metadata.csv` and the θ arrays of the released
events; it needs only the ISMN archive. Its sensor pairs are written to
`outputs/runs/check_ismn/sensor_pairs.parquet`.

## Table schemas

Schemas are asserted at write time ([`schemas.py`](src/drydown_q/schemas.py)).
Units are in the column names. Filters add columns and never drop rows.

### `sensors`: one soil moisture sensor (from `prepare`)

| Column | Type | Meaning |
| --- | --- | --- |
| `sensor_id` | str | `network/station/<model>_<probe>_soil_moisture_<depth_from>_<depth_to>` for a whole probe record (`prepare.sensor_unit = probe`, joining the archive's deployment-period files `<model>_<probe>_<period>`), or the ISMN file's sensor name (`sensor_unit = file`) |
| `station_id` | str | `network/station` |
| `network` | str | ISMN network |
| `lat_deg`, `lon_deg` | float | station coordinates [°] |
| `depth_top_m`, `depth_bottom_m` | float | sensor depth range [m] |
| `sand_frac`, `silt_frac`, `clay_frac` | float | ISMN metadata texture fractions [0–1] |
| `texture_class` | str | USDA texture class (rules in `data/tables/usda_texture_rules.csv`) |
| `chirps_offset_km`, `dpet_offset_km` | float | distance from station to the centre of the grid cell used [km] |
| `chirps_cell_is_land`, `dpet_cell_is_land` | bool | the cell used has valid land data |

### `station_day`: one sensor on one UTC day (from `prepare`)

| Column | Type | Meaning |
| --- | --- | --- |
| `sensor_id` | str | |
| `date` | date | UTC calendar day |
| `theta_m3m3` | float | daily mean volumetric soil moisture [m³ m⁻³] of kept hourly values |
| `n_valid_hours` | int | hourly values with a kept ISMN flag |
| `precip_mm`, `pet_mm` | float | CHIRPS precipitation and dPET potential ET [mm d⁻¹] |
| `precip_60d_mm`, `pet_60d_mm` | float | trailing sums over `covariates.ai_window_days` (60) ending on `date`, computed on the full forcing record [mm] |
| `lai_m2m2` | float | MODIS LAI on `date` (latest composite) [m² m⁻²] |
| `igbp_class` | str | MODIS MCD12Q1 IGBP class of that year (e.g. `GRA`); empty when unavailable |
| `tree_cover_pct` | float | MOD44B tree cover of that year [%] |
| `precip_insitu_mm` | float | optional: ISMN gauge precipitation, daily sum of kept records, NaN when none [mm d⁻¹] |

### `sensor_constants`: one sensor (analysis step 1, depends on `soil`)

| Column | Type | Meaning |
| --- | --- | --- |
| `sensor_id` | str | |
| `theta_fc_m3m3` | float | field capacity [m³ m⁻³] |
| `theta_fc_source` | str | `clapp_hornberger` or `max_sm` (fallback) |
| `max_sm_m3m3`, `min_sm_m3m3` | float | quantile maximum and minimum of daily θ [m³ m⁻³] |
| `theta_w_m3m3` | float | wilting point used by the models [m³ m⁻³] |
| `theta_w_source` | str | rule applied: `record_min`, `quantile`, `seasonal_min` or `record_min_fallback` |
| `dz_m` | float | layer depth Δz [m] |

### `events`: one drydown

| Column | Type | Meaning |
| --- | --- | --- |
| `event_id` | str | `<sensor_id>/<start_date>` |
| `sensor_id` | str | |
| `start_date`, `end_date` | date | first and last day |
| `duration_days` | int | `end_date − start_date` [d] |
| `n_obs` | int | non-missing days fitted |
| `n_days_above_fc` | int | days in the event with raw θ > θ_fc |
| `end_reason` | str | `decline_below_threshold`, `gap`, `record_end` or (with `precip_check = whole_event`) `rain` |

### `precip_sources`: one sensor (step 2)

`sensor_id`, `source` (`chirps` or `insitu_then_chirps`), `reason` (why a gauge was not used),
`gauge_wet_share`, `chirps_wet_share` (share of common days with P > `precip_max_mm_day`).

### `event_days`: one day in one event

Every day from start to end is a row, including days with missing θ (the fit drops
them; PET_max covers all event days).

| Column | Type | Meaning |
| --- | --- | --- |
| `event_id` | str | |
| `t_days` | int | days since the start |
| `theta_m3m3` | float | unmasked daily θ [m³ m⁻³]; NaN on missing days |
| `pet_mm` | float | dPET [mm d⁻¹] |

### `fits`: one event × one model

`event_id`, `model`, each fitted or fixed parameter with its unit in the name
(e.g. `k_per_day`, `q`, `theta_star_m3m3`, `tau_days`, `theta_w_m3m3`) and its
standard error `<param>_se`, `r2`, `rmse_m3m3`, `fit_status`, `fit_message`, and a
bool `at_bound_<param>` per bounded parameter.

### `covariates`: the `analysis` table before the filter flags (step 4)

### `analysis`: one event with its fit and covariates

The `fits` columns of the q model, plus `sensor_id`, `station_id`, `ai_60d`,
`lai_m2m2`, `veg_class`, `frac_range`, one bool `pass_<rule>` per filter, and
`in_final_set` (AND of the active rules).

### `bins`: class × AI₆₀ bin × LAI bin

`veg_class`, `ai_bin`, `lai_bin` (category codes), `ai_mid`, `lai_mid`, `n_events`,
`q_median`, `used_in_fit`.

### `surfaces`: one class

`veg_class`, `a`, `b`, `c`, `d` and `a_se` … `d_se`, `r2_bins`, `n_bins`, `n_events`
(summed over used bins), `fit_status` (`ok`, `too_few_bins` or `failed: <message>`).

Each run also writes `outputs/runs/<run>/summary.json` with its headline numbers.

### `summary`: one sweep configuration

`run`, `changed_key`, `changed_value`, `status`, `message`, `events_final`,
`stations_final`, `events_surface`, and per class `median_q_<class>`, `a_<class>` …
`d_<class>`, `r2_bins_<class>`, `n_bins_<class>`, `dq_dai_at_median_lai_<class>`,
`dq_dai_sign_<class>`; plus `herb_minus_wood_q`.
