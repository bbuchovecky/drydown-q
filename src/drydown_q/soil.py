"""Per-sensor soil constants: texture class, field capacity theta_fc, wilting
point theta_w, max_sm/min_sm, and layer depth dz.

Tables: ``data/tables/usda_texture_rules.csv`` (texture triangle, transcribed
from Morgan's ``get_soil_texture``) and ``data/tables/clapp_hornberger_1978.csv``.
"""

import numpy as np
import pandas as pd

from drydown_q.config import TABLES_DIR, Soil

NOT_AVAILABLE = "unknown"


def texture_class(sand_frac: pd.Series, silt_frac: pd.Series, clay_frac: pd.Series) -> pd.Series:
    """USDA texture class from sand, silt and clay mass fractions [0-1].

    The rules are applied in table order and a later matching rule overwrites
    an earlier one, exactly as in Morgan's code. Missing or out-of-range
    fractions give ``"unknown"``.
    """
    rules = pd.read_csv(TABLES_DIR / "usda_texture_rules.csv").sort_values("order")
    fractions = {"sand": sand_frac.to_numpy(float), "silt": silt_frac.to_numpy(float),
                 "clay": clay_frac.to_numpy(float)}
    out = np.full(len(sand_frac), NOT_AVAILABLE, dtype=object)
    for rule in rules.itertuples():
        match = np.ones(len(out), dtype=bool)
        for part, values in fractions.items():
            lo, hi = getattr(rule, f"{part}_min"), getattr(rule, f"{part}_max")
            if pd.notna(lo):
                match &= values >= lo
            if pd.notna(hi):
                match &= values <= hi
        out[match] = rule.texture_class
    valid = np.all([(v >= 0) & (v <= 1) for v in fractions.values()], axis=0)
    out[~valid] = NOT_AVAILABLE
    return pd.Series(out, index=sand_frac.index, dtype=str)


def field_capacity(texture: pd.Series, cfg: Soil) -> pd.Series:
    """theta_fc [m3 m-3] from Clapp & Hornberger (1978) at psi_fc.

        psi_s  = -psi_s_cm / 100 * rho_w * g / 1000                 [kPa]
        theta_fc = min( porosity * (psi_fc / psi_s)^(-1/b), porosity )
        rounded to ``theta_fc_round_m3m3`` (Morgan rounds to 0.01)

    psi_s_cm is the saturated tension head [cm] (positive), b the pore-size
    index [-], psi_fc = ``psi_fc_kpa`` [kPa]. Texture classes without C&H
    parameters (``silt``, ``unknown``) give NaN; the caller falls back to max_sm.
    """
    ch = pd.read_csv(TABLES_DIR / "clapp_hornberger_1978.csv").set_index("texture_class")
    params = ch.reindex(texture.to_numpy())
    psi_s_kpa = -params["psi_s_cm"].to_numpy() / 100 * cfg.rho_water_kg_m3 * cfg.g_m_s2 / 1000
    porosity = params["porosity_m3m3"].to_numpy()
    theta = porosity * (cfg.psi_fc_kpa / psi_s_kpa) ** (-1 / params["b"].to_numpy())
    theta = np.fmin(_round_to(theta, cfg.theta_fc_round_m3m3), porosity)
    return pd.Series(theta, index=texture.index, dtype=float)


def _round_to(values: np.ndarray, step: float | None) -> np.ndarray:
    """Round to a power-of-ten step with Python's round(), as Morgan's code does."""
    if step is None:
        return values
    decimals = int(round(-np.log10(step)))
    return np.array([round(v, decimals) if np.isfinite(v) else v for v in values], dtype=float)


def wilting_point(theta_daily: pd.Series, cfg: Soil) -> float:
    """theta_w [m3 m-3] of one sensor from its daily theta.

    ``record_min``: minimum over the sensor record (Morgan: theta_w = min_sm).
    ``quantile``: the ``theta_w_quantile`` quantile of daily theta.
    """
    if cfg.theta_w_rule == "record_min":
        return float(theta_daily.min())
    if cfg.theta_w_rule == "quantile":
        if cfg.theta_w_quantile is None:
            raise ValueError("soil.theta_w_rule = quantile needs soil.theta_w_quantile")
        return float(theta_daily.quantile(cfg.theta_w_quantile))
    raise ValueError(f"soil.theta_w_rule = {cfg.theta_w_rule!r} is not defined yet")


def layer_depth(sensors: pd.DataFrame, cfg: Soil) -> pd.Series:
    """dz [m], the soil layer a sensor represents (``dz_rule``).

    ``depth_bottom``: dz = depth_bottom_m (Morgan's run used the sensor's depth_to).
    ``layer_thickness``: Morgan's ``calc_dz``: at each station, sort the distinct
        sensor bottom depths z_1 < ... < z_n; layer boundaries are the midpoints
        (z_i + z_{i+1})/2 and the last depth; dz_i = boundary_i - boundary_{i-1},
        with boundary_0 = 0.
    ``fixed``: dz = ``dz_fixed_m`` for every sensor.
    """
    if cfg.dz_rule == "depth_bottom":
        return sensors["depth_bottom_m"].astype(float)
    if cfg.dz_rule == "fixed":
        if cfg.dz_fixed_m is None:
            raise ValueError("soil.dz_rule = fixed needs soil.dz_fixed_m")
        return pd.Series(float(cfg.dz_fixed_m), index=sensors.index)
    dz = pd.Series(np.nan, index=sensors.index)
    for _, group in sensors.groupby("station_id"):
        depths = np.sort(group["depth_bottom_m"].unique())
        bounds = np.append((depths[:-1] + depths[1:]) / 2, depths[-1])
        thickness = np.diff(np.insert(bounds, 0, 0))
        dz[group.index] = group["depth_bottom_m"].map(dict(zip(depths, thickness)))
    return dz


def sensor_constants(sensors: pd.DataFrame, station_day: pd.DataFrame, cfg: Soil) -> pd.DataFrame:
    """The ``sensor_constants`` table (analysis step 1).

        max_sm = ``max_sm_quantile`` quantile of daily theta     [m3 m-3]
        min_sm = minimum of daily theta                          [m3 m-3]
        theta_fc = Clapp & Hornberger (``theta_fc_rule = clapp_hornberger``),
                   or max_sm when the texture has no C&H parameters
                   (``theta_fc_source = max_sm``)
    """
    theta = station_day.groupby("sensor_id")["theta_m3m3"]
    stats = pd.DataFrame({"max_sm_m3m3": theta.quantile(cfg.max_sm_quantile),
                          "min_sm_m3m3": theta.min()})
    stats["theta_w_m3m3"] = [wilting_point(g, cfg) for _, g in theta]
    out = sensors[["sensor_id"]].join(stats, on="sensor_id")
    if cfg.theta_fc_rule == "clapp_hornberger":
        fc = field_capacity(sensors["texture_class"], cfg)
    else:
        fc = pd.Series(np.nan, index=sensors.index)
    out["theta_fc_source"] = np.where(fc.notna(), cfg.theta_fc_rule, "max_sm")
    out["theta_fc_m3m3"] = fc.fillna(out["max_sm_m3m3"])
    out["dz_m"] = layer_depth(sensors, cfg)
    out["theta_fc_source"] = out["theta_fc_source"].astype(str)
    return out[["sensor_id", "theta_fc_m3m3", "theta_fc_source", "max_sm_m3m3",
                "min_sm_m3m3", "theta_w_m3m3", "dz_m"]].reset_index(drop=True)
