"""Event filters. Flag, don't drop: each rule adds a boolean ``pass_<rule>``
column; ``in_final_set`` is the AND of the active rules. A rule whose setting
is null is inactive and its column is all True.

Rules, in cascade order (all read the ``filters`` config section). A NaN in a
compared column fails the rule, as in Morgan's pandas comparisons.

    r2                       r2 >= r2_min                         (q model fit)
    duration                 duration_days[0] <= duration <= duration_days[1]
    q_min                    q > q_min
    frac_range               frac_range > frac_range_min
    mean_depth               (depth_top + depth_bottom) / 2 <= mean_depth_max_m
    dz                       dz <= dz_max_m
    theta_star_minus_min_sm  theta_star - min_sm >= theta_star_minus_min_sm_min
    theta_star_minus_w       theta_star - theta_w >= theta_star_minus_w_min
    abs_lat                  |lat| <= abs_lat_max_deg                (if set)
    class                    veg_class in classes
    lai                      LAI >= lai_min_m2m2 (lai_min_inclusive) or LAI > lai_min_m2m2
"""

import pandas as pd

from drydown_q.config import Filters

QUALITY_RULES = ["r2", "duration", "q_min", "frac_range", "mean_depth", "dz",
                 "theta_star_minus_min_sm", "theta_star_minus_w", "abs_lat"]
RULES = QUALITY_RULES + ["class", "lai"]


def flag_filters(analysis: pd.DataFrame, cfg: Filters) -> pd.DataFrame:
    """Return ``analysis`` with ``pass_<rule>`` for every rule and ``in_final_set``."""
    a = analysis.copy()
    lo, hi = cfg.duration_days
    a["pass_r2"] = a["r2"] >= cfg.r2_min
    a["pass_duration"] = (a["duration_days"] >= lo) & (a["duration_days"] <= hi)
    a["pass_q_min"] = a["q"] > cfg.q_min
    a["pass_frac_range"] = a["frac_range"] > cfg.frac_range_min
    a["pass_mean_depth"] = a["mean_depth_m"] <= cfg.mean_depth_max_m
    a["pass_dz"] = a["dz_m"] <= cfg.dz_max_m
    a["pass_theta_star_minus_min_sm"] = (
        a["theta_star_m3m3"] - a["min_sm_m3m3"] >= cfg.theta_star_minus_min_sm_min)
    a["pass_theta_star_minus_w"] = (
        a["theta_star_m3m3"] - a["theta_w_m3m3"] >= cfg.theta_star_minus_w_min)
    if cfg.abs_lat_max_deg is None:
        a["pass_abs_lat"] = True
    else:
        a["pass_abs_lat"] = a["lat_deg"].abs() <= cfg.abs_lat_max_deg
    a["pass_class"] = a["veg_class"].isin(cfg.classes)
    if cfg.lai_min_inclusive:
        a["pass_lai"] = a["lai_m2m2"] >= cfg.lai_min_m2m2
    else:
        a["pass_lai"] = a["lai_m2m2"] > cfg.lai_min_m2m2
    for rule in RULES:
        a[f"pass_{rule}"] = a[f"pass_{rule}"].astype(bool)
    a["in_final_set"] = a[[f"pass_{r}" for r in RULES]].all(axis=1)
    return a


def cascade(analysis: pd.DataFrame) -> pd.DataFrame:
    """Events and stations remaining after each rule, applied cumulatively in
    ``RULES`` order. Row 0 is the input; ``removed`` is the events dropped by
    that rule given all earlier rules."""
    keep = pd.Series(True, index=analysis.index)
    rows = [{"after": "input", "events": len(analysis),
             "stations": analysis["station_id"].nunique(), "removed": 0}]
    for rule in RULES:
        before = int(keep.sum())
        keep &= analysis[f"pass_{rule}"]
        rows.append({"after": rule, "events": int(keep.sum()),
                     "stations": analysis.loc[keep, "station_id"].nunique(),
                     "removed": before - int(keep.sum())})
    return pd.DataFrame(rows)
