"""Fixed schemas of the tables each stage writes (documented in README.md).

Each schema maps a required column to a kind: ``str``, ``float``, ``int``,
``bool`` or ``date`` (datetime64, UTC calendar day). ``extra`` lists glob
patterns, with a kind, for columns that vary with the configuration (model
parameters, filter flags). Any other column is an error, so a table cannot
silently grow or lose columns.
"""

import fnmatch

import pandas as pd

SCHEMAS = {
    "sensors": {
        "required": {
            "sensor_id": "str", "station_id": "str", "network": "str",
            "lat_deg": "float", "lon_deg": "float",
            "depth_top_m": "float", "depth_bottom_m": "float",
            "sand_frac": "float", "silt_frac": "float", "clay_frac": "float",
            "texture_class": "str",
            "chirps_offset_km": "float", "chirps_cell_is_land": "bool",
            "dpet_offset_km": "float", "dpet_cell_is_land": "bool",
        },
        "extra": {},
    },
    "station_day": {
        "required": {
            "sensor_id": "str", "date": "date", "theta_m3m3": "float",
            "n_valid_hours": "int", "precip_mm": "float", "pet_mm": "float",
            "precip_60d_mm": "float", "pet_60d_mm": "float", "lai_m2m2": "float",
            "igbp_class": "str", "tree_cover_pct": "float",
        },
        "extra": {"precip_insitu_mm": "float"},   # ISMN gauge precipitation [mm d-1], where available
    },
    # Step 1 output: per-sensor constants that depend on the `soil` config section.
    "sensor_constants": {
        "required": {
            "sensor_id": "str", "theta_fc_m3m3": "float", "theta_fc_source": "str",
            "max_sm_m3m3": "float", "min_sm_m3m3": "float", "theta_w_m3m3": "float",
            "dz_m": "float",
        },
        "extra": {},
    },
    "events": {
        "required": {
            "event_id": "str", "sensor_id": "str", "start_date": "date",
            "end_date": "date", "duration_days": "int", "n_obs": "int",
            "n_days_above_fc": "int", "end_reason": "str",
        },
        "extra": {},
    },
    "event_days": {
        "required": {"event_id": "str", "t_days": "int", "theta_m3m3": "float",
                     "pet_mm": "float"},
        "extra": {},
    },
    "fits": {
        "required": {"event_id": "str", "model": "str", "r2": "float",
                     "rmse_m3m3": "float", "fit_status": "str", "fit_message": "str"},
        "extra": {"at_bound_*": "bool", "*_se": "float", "*": "float"},
    },
    # Step 4 output: the analysis table before the filters add their flags.
    "covariates": {
        "required": {"event_id": "str", "sensor_id": "str", "station_id": "str",
                     "ai_60d": "float", "lai_m2m2": "float", "frac_range": "float"},
        "extra": {"at_bound_*": "bool", "*": "any"},
    },
    "analysis": {
        "required": {"event_id": "str", "sensor_id": "str", "station_id": "str",
                     "ai_60d": "float", "lai_m2m2": "float", "veg_class": "str",
                     "frac_range": "float", "in_final_set": "bool"},
        "extra": {"pass_*": "bool", "at_bound_*": "bool", "*": "any"},
    },
    "bins": {
        "required": {"veg_class": "str", "ai_bin": "int", "lai_bin": "int",
                     "ai_mid": "float", "lai_mid": "float", "n_events": "int",
                     "q_median": "float", "used_in_fit": "bool"},
        "extra": {},
    },
    "surfaces": {
        "required": {"veg_class": "str", "a": "float", "b": "float", "c": "float",
                     "d": "float", "a_se": "float", "b_se": "float", "c_se": "float",
                     "d_se": "float", "r2_bins": "float", "n_bins": "int",
                     "n_events": "int", "fit_status": "str"},
        "extra": {},
    },
    # One row per sweep configuration; per-class columns are <quantity>_<class>.
    "summary": {
        "required": {"run": "str", "changed_key": "str", "changed_value": "str", "status": "str",
                     "events_final": "float", "stations_final": "float", "events_surface": "float"},
        "extra": {"median_q_*": "float", "herb_minus_wood_q": "float", "a_*": "float", "b_*": "float",
                  "c_*": "float", "d_*": "float", "r2_bins_*": "float", "n_bins_*": "float",
                  "dq_dai_sign_*": "float", "dq_dai_at_median_lai_*": "float", "message": "str"},
    },
}


def _kind_ok(series: pd.Series, kind: str) -> bool:
    if kind == "any":
        return True
    if kind == "str":
        return pd.api.types.is_string_dtype(series) or series.dtype == object
    if kind == "float":
        return pd.api.types.is_float_dtype(series)
    if kind == "int":
        return pd.api.types.is_integer_dtype(series)
    if kind == "bool":
        return pd.api.types.is_bool_dtype(series)
    if kind == "date":
        return pd.api.types.is_datetime64_dtype(series)
    raise ValueError(f"unknown kind {kind!r}")


def assert_schema(df: pd.DataFrame, table: str) -> None:
    """Raise ``AssertionError`` if ``df`` does not follow ``SCHEMAS[table]``."""
    schema = SCHEMAS[table]
    missing = [c for c in schema["required"] if c not in df.columns]
    assert not missing, f"{table}: missing columns {missing}"
    problems = []
    for column in df.columns:
        if column in schema["required"]:
            kind = schema["required"][column]
        else:
            kind = next((k for pattern, k in schema["extra"].items()
                         if fnmatch.fnmatch(column, pattern)), None)
            if kind is None:
                problems.append(f"unexpected column {column!r}")
                continue
        if not _kind_ok(df[column], kind):
            problems.append(f"{column!r} has dtype {df[column].dtype}, expected {kind}")
    assert not problems, f"{table}: " + "; ".join(problems)
