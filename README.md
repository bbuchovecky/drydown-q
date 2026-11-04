# drydown-q

A readable reimplementation of the soil moisture drydown analysis of
Morgan et al. (2025), *Nature Ecology & Evolution*,
[doi:10.1038/s41559-025-02810-8](https://doi.org/10.1038/s41559-025-02810-8).

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

## Table schemas

Schemas are asserted at write time ([`schemas.py`](src/drydown_q/schemas.py)).
Units are in the column names. Filters add columns and never drop rows.

### `sensors`: one soil moisture sensor (from `prepare`)

| Column | Type | Meaning |
| --- | --- | --- |
| `sensor_id` | str | `network/station/<variable>_<depth_from>_<depth_to>_<instrument>` |
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
| `precip_60d_mm`, `pet_60d_mm` | float | trailing 60-day sums ending on `date` [mm] |
| `lai_m2m2` | float | MODIS LAI on `date` (latest composite) [m² m⁻²] |
| `igbp_class` | str | MODIS MCD12Q1 IGBP class of that year (e.g. `GRA`) |
| `tree_cover_pct` | float | MOD44B tree cover of that year [%] |

### `sensor_constants`: one sensor (analysis step 1, depends on `soil`)

| Column | Type | Meaning |
| --- | --- | --- |
| `sensor_id` | str | |
| `theta_fc_m3m3` | float | field capacity [m³ m⁻³] |
| `theta_fc_source` | str | `clapp_hornberger` or `max_sm` (fallback) |
| `max_sm_m3m3`, `min_sm_m3m3` | float | quantile maximum and minimum of daily θ [m³ m⁻³] |
| `theta_w_m3m3` | float | wilting point used by the models [m³ m⁻³] |
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
| `end_reason` | str | `decline_below_threshold`, `gap` or `record_end` |

### `event_days`: one day in one event

| Column | Type | Meaning |
| --- | --- | --- |
| `event_id` | str | |
| `t_days` | int | days since the start |
| `theta_m3m3` | float | unmasked daily θ [m³ m⁻³] |
| `pet_mm` | float | dPET [mm d⁻¹] |

### `fits`: one event × one model

`event_id`, `model`, each fitted or fixed parameter with its unit in the name
(e.g. `k_per_day`, `q`, `theta_star_m3m3`, `tau_days`, `theta_w_m3m3`) and its
standard error `<param>_se`, `r2`, `rmse_m3m3`, `fit_status`, `fit_message`, and a
bool `at_bound_<param>` per bounded parameter.

### `analysis`: one event with its fit and covariates

The `fits` columns of the q model, plus `sensor_id`, `station_id`, `ai_60d`,
`lai_m2m2`, `veg_class`, `frac_range`, one bool `pass_<rule>` per filter, and
`in_final_set` (AND of the active rules).

### `bins`: class × AI₆₀ bin × LAI bin

`veg_class`, `ai_bin`, `lai_bin` (category codes), `ai_mid`, `lai_mid`, `n_events`,
`q_median`, `used_in_fit`.

### `surfaces`: one class

`veg_class`, `a`, `b`, `c`, `d` and `a_se` … `d_se`, `r2_bins`, `n_bins`, `n_events`.

### `summary`: one sweep run

Changed settings, counts, median q by class, coefficients, R², sign of ∂q/∂AI₆₀ at
the class median LAI.
