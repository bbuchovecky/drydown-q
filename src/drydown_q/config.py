"""Load, merge and validate the configuration.

Every scientific setting lives in ``config/default.yml``. A run file
(``config/runs/<name>.yml``) overrides only the keys it lists, and the paths
file (``config/paths.yml``, git-ignored) holds machine-specific locations.
The merge order is default -> run -> paths.

The dataclasses below are the schema: one frozen dataclass per section, with
field types but no default values (defaults live only in ``default.yml``).
Unknown keys, missing keys, wrong types and values outside a ``Literal`` are
rejected. This module holds no scientific numbers.
"""

import dataclasses
import hashlib
import itertools
import json
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
TABLES_DIR = REPO_ROOT / "data" / "tables"
SRC_DIR = REPO_ROOT / "src"

Number = int | float


@dataclass(frozen=True)
class Prepare:
    sensor_unit: Literal["probe", "file"]
    ismn_flags_kept: list[str]
    day_definition: Literal["utc"]
    min_valid_hours: int
    grid_cell_rule: Literal["nearest", "nearest_land_within_km"]
    max_cell_offset_km: Number | None
    require_land_cell: bool
    lai_product: Literal["MOD15A2H.061", "MCD15A2H.061"]
    lai_qc: Literal["none"]
    lai_keep_below: Number
    lai_fill_max_days: int | None
    pet_valid_max_mm_day: Number
    earth_radius_km: Number


@dataclass(frozen=True)
class Soil:
    theta_fc_rule: Literal["clapp_hornberger", "max_sm"]
    psi_fc_kpa: Number
    theta_fc_round_m3m3: Number | None
    rho_water_kg_m3: Number
    g_m_s2: Number
    max_sm_quantile: Number
    theta_w_rule: Literal["record_min", "quantile", "seasonal_min"]
    theta_w_quantile: Number | None
    dz_rule: Literal["depth_bottom", "layer_thickness", "fixed"]
    dz_fixed_m: Number | None


@dataclass(frozen=True)
class Events:
    mask_above_fc: bool
    start_min_decline_mm_day: Number
    end_min_decline_mm_day: Number
    min_decline_days: int
    precip_max_mm_day: Number
    precip_check: Literal["start_window", "whole_event"]
    precip_missing_is_dry: bool
    precip_source: Literal["chirps", "insitu_then_chirps"]
    gap_rule: Literal["end_event"]
    exclude_snow_frozen: bool


@dataclass(frozen=True)
class Fitting:
    models: list[Literal["exponential", "q_stage2", "q_piecewise"]]
    free_params_q: list[Literal["k", "q", "theta_star"]]
    theta_0_rule: Literal["first_obs"]
    q_bounds: list[Number]
    q_start: Number
    theta_star_bounds: list[str]
    k_upper: Literal["max_daily_pet"]
    k_start_fraction: Number
    pet_default_mm_day: Number
    exp_theta_w_bounds: list[str]
    exp_tau_bounds: list[Number]
    exp_tau_start: Number


@dataclass(frozen=True)
class Covariates:
    ai_window_days: int
    ai_window_includes_start: bool
    ai_window_min_valid_days: int
    lai_statistic: Literal["start_day", "event_max", "event_mean"]
    class_scheme: str
    landcover_year: Literal["event_year"]


@dataclass(frozen=True)
class Filters:
    r2_min: Number
    duration_days: list[int]
    q_min: Number
    frac_range_min: Number
    mean_depth_max_m: Number
    dz_max_m: Number
    theta_star_minus_min_sm_min: Number
    theta_star_minus_w_min: Number
    classes: list[str]
    lai_min_m2m2: Number
    lai_min_inclusive: bool
    abs_lat_max_deg: Number | None


@dataclass(frozen=True)
class Binning:
    ai_edges: list[Number]
    lai_edges: list[Number]
    bins_right_closed: bool
    open_bin_upper_for_mid: dict[str, Number]
    lai_min_inclusive_surface: bool
    min_events_per_bin: int
    statistic: Literal["median"]
    bin_weights: Literal["equal"]
    surface_form: Literal["a*ai + b*lai + c*ai*lai + d"]


@dataclass(frozen=True)
class Runtime:
    n_jobs: int
    log_level: Literal["DEBUG", "INFO", "WARNING"]


@dataclass(frozen=True)
class Paths:
    ismn_archive: str | None
    chirps_dir: str | None
    dpet_dir: str | None
    appeears_dir: str | None
    morgan_repo: str | None
    outputs_dir: str
    prepared_dir: str | None  # where prepare.py writes sensors/station_day; null = <outputs_dir>/prepared


@dataclass(frozen=True)
class Config:
    run_name: str
    prepare: Prepare
    soil: Soil
    events: Events
    fitting: Fitting
    covariates: Covariates
    filters: Filters
    binning: Binning
    runtime: Runtime
    paths: Paths


SECTIONS = {
    "prepare": Prepare, "soil": Soil, "events": Events, "fitting": Fitting,
    "covariates": Covariates, "filters": Filters, "binning": Binning, "runtime": Runtime,
}


class ConfigError(ValueError):
    """Raised for unknown keys, missing keys or values of the wrong type."""


def _matches(value, annotation) -> bool:
    """True if ``value`` (parsed from YAML) is an instance of ``annotation``."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if annotation is type(None):
        return value is None
    if origin in (typing.Union, types.UnionType):
        return any(_matches(value, a) for a in args)
    if origin is Literal:
        return value in args
    if origin is list:
        return isinstance(value, list) and all(_matches(v, args[0]) for v in value)
    if origin is dict:
        return isinstance(value, dict) and all(
            _matches(k, args[0]) and _matches(v, args[1]) for k, v in value.items())
    if annotation is int:  # YAML booleans are ints in Python; reject them here
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, float)
    return isinstance(value, annotation)


def _build(cls, values: dict, where: str):
    fields = {f.name: f.type for f in dataclasses.fields(cls)}
    unknown = set(values) - set(fields)
    missing = set(fields) - set(values)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")
    if missing:
        raise ConfigError(f"{where}: missing key(s) {sorted(missing)}")
    for name, annotation in fields.items():
        if not _matches(values[name], annotation):
            raise ConfigError(
                f"{where}.{name} = {values[name]!r} does not match type {annotation}")
    return cls(**values)


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def merge(default: dict, override: dict, where: str) -> dict:
    """Return ``default`` with the keys of ``override`` replaced, section by section.

    An override may only name sections and keys that exist in ``default``.
    """
    merged = {section: dict(values) for section, values in default.items()}
    for section, values in override.items():
        if section not in merged:
            raise ConfigError(f"{where}: unknown section {section!r}")
        if not isinstance(values, dict):
            raise ConfigError(f"{where}: section {section!r} must be a mapping")
        for key, value in values.items():
            if key not in merged[section]:
                raise ConfigError(f"{where}: unknown key {section}.{key}")
            merged[section][key] = value
    return merged


def load_config(run: str, paths_file: Path | None = None,
                overrides: dict | None = None) -> Config:
    """Load default.yml, apply ``config/runs/<run>.yml``, extra ``overrides``, then paths.

    ``overrides`` (same nested shape as a run file) is how sweep.py varies settings.
    Relative paths in the paths file are resolved against the repository root.
    """
    default = _read_yaml(CONFIG_DIR / "default.yml")
    run_file = CONFIG_DIR / "runs" / f"{run}.yml"
    if not run_file.exists():
        raise ConfigError(f"no run file {run_file}")
    merged = merge(default, _read_yaml(run_file), str(run_file))
    if overrides:
        merged = merge(merged, overrides, "overrides")

    paths_file = Path(paths_file) if paths_file else CONFIG_DIR / "paths.yml"
    if not paths_file.exists():
        raise ConfigError(f"no paths file {paths_file}; copy config/paths.example.yml")
    paths = _read_yaml(paths_file)
    for key, value in paths.items():
        if isinstance(value, str) and not Path(value).expanduser().is_absolute():
            paths[key] = str(REPO_ROOT / value)
        elif isinstance(value, str):
            paths[key] = str(Path(value).expanduser())
    paths.setdefault("prepared_dir", None)
    if paths["prepared_dir"] is None and "outputs_dir" in paths:
        paths["prepared_dir"] = str(Path(paths["outputs_dir"]) / "prepared")

    sections = {name: _build(cls, merged.get(name, {}), name) for name, cls in SECTIONS.items()}
    return Config(run_name=run, paths=_build(Paths, paths, str(paths_file)), **sections)


def load_sweep(name: str) -> tuple[str, list[tuple[str, dict]]]:
    """Read ``config/sweeps/<name>.yml``; return ``(base_run, configurations)``.

    A sweep file has ``base_run``, ``mode`` and ``changes`` (``"section.key":
    [values]``). ``one_at_a_time``: each key takes each listed value with all
    else at the base run; ``grid``: every combination. Each configuration is
    ``(label, overrides)``; the base run itself is ``("baseline", {})`` and any
    configuration equal to it is dropped. Every configuration is validated.
    """
    path = CONFIG_DIR / "sweeps" / f"{name}.yml"
    if not path.exists():
        raise ConfigError(f"no sweep file {path}")
    sweep = _read_yaml(path)
    unknown = set(sweep) - {"base_run", "mode", "changes"}
    if unknown or not {"base_run", "mode", "changes"} <= set(sweep):
        raise ConfigError(f"{path}: needs exactly base_run, mode, changes (got {sorted(sweep)})")
    if sweep["mode"] not in ("one_at_a_time", "grid"):
        raise ConfigError(f"{path}: mode must be one_at_a_time or grid")
    default = _read_yaml(CONFIG_DIR / "default.yml")
    base = merge(default, _read_yaml(CONFIG_DIR / "runs" / f"{sweep['base_run']}.yml"), sweep["base_run"])
    changes = sweep["changes"]
    if sweep["mode"] == "one_at_a_time":
        combos = [{key: value} for key, values in changes.items() for value in values]
    else:
        keys = list(changes)
        combos = [dict(zip(keys, values)) for values in itertools.product(*changes.values())]
    configs, seen = [("baseline", {})], {config_hash(base)}
    for combo in combos:
        overrides = {}
        for dotted, value in combo.items():
            section, _, key = dotted.partition(".")
            overrides.setdefault(section, {})[key] = value
        merged = merge(base, overrides, f"{path}: {combo}")
        for section, cls in SECTIONS.items():
            _build(cls, merged[section], f"{path}: {section}")
        if config_hash(merged) in seen:
            continue
        seen.add(config_hash(merged))
        label = "__".join(f"{k}={_label(v)}" for k, v in combo.items())
        configs.append((label, overrides))
    return sweep["base_run"], configs


def _label(value) -> str:
    return "-".join(map(str, value)) if isinstance(value, list) else str(value)


def as_dict(obj) -> dict:
    """Plain-dict view of a config or a config section (for hashing and writing)."""
    return dataclasses.asdict(obj)


def config_hash(*parts) -> str:
    """Short SHA-256 of config sections, dicts or strings, in canonical JSON form."""
    canonical = [as_dict(p) if dataclasses.is_dataclass(p) else p for p in parts]
    text = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def code_hash(src_dir: Path = SRC_DIR) -> str:
    """Short SHA-256 of the contents of every .py file under ``src/``."""
    digest = hashlib.sha256()
    for path in sorted(Path(src_dir).rglob("*.py")):
        digest.update(str(path.relative_to(src_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def differences_from_default(cfg: Config) -> dict[str, tuple]:
    """``{"section.key": (default, actual)}`` for every scientific setting that differs."""
    default = _read_yaml(CONFIG_DIR / "default.yml")
    diffs = {}
    for section in SECTIONS:
        actual = as_dict(getattr(cfg, section))
        for key, value in actual.items():
            if default[section][key] != value:
                diffs[f"{section}.{key}"] = (default[section][key], value)
    return diffs


def write_resolved(cfg: Config, path: Path) -> None:
    """Write the fully merged config (all sections and paths) as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(as_dict(cfg), f, sort_keys=False)
