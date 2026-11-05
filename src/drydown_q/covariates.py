"""Event covariates (antecedent aridity AI_60, LAI, land cover) and the
analysis table that the filters and bins work on."""

import pandas as pd

from drydown_q.config import TABLES_DIR, Covariates
from drydown_q.gridded import window_sum


def assign_class(igbp_class: pd.Series, scheme: str) -> pd.Series:
    """Map MODIS IGBP codes (e.g. ``GRA``) to vegetation classes with a scheme
    from ``data/tables/class_schemes.csv``. Codes absent from the table map to NaN."""
    schemes = pd.read_csv(TABLES_DIR / "class_schemes.csv")
    rows = schemes[schemes["scheme"] == scheme]
    if rows.empty:
        raise ValueError(f"class scheme {scheme!r} not in class_schemes.csv")
    return igbp_class.map(rows.set_index("igbp_class")["veg_class"])


def event_covariates(events: pd.DataFrame, station_day: pd.DataFrame,
                     cfg: Covariates) -> pd.DataFrame:
    """One row per event: ``ai_60d``, ``precip_window_mm``, ``pet_window_mm``,
    ``lai_m2m2``, ``igbp_class``, ``tree_cover_pct``.

        AI_60 = sum(P) / sum(PET) over the ``ai_window_days`` days ending on the
                start day (``ai_window_includes_start``) or the day before.  [-]

    A window with fewer than ``ai_window_min_valid_days`` valid days of P or PET
    gives NaN. LAI is the value on the start day (``lai_statistic = start_day``;
    station_day already holds the latest 8-day composite), or the max/mean over
    the event days. Land cover and tree cover are those of the start year
    (``landcover_year = event_year``; station_day carries the yearly class).
    """
    rows = []
    for sensor_id, ev in events.groupby("sensor_id", sort=False):
        day = station_day[station_day["sensor_id"] == sensor_id].set_index("date").sort_index()
        p_sum = window_sum(day["precip_mm"], cfg.ai_window_days, cfg.ai_window_min_valid_days)
        e_sum = window_sum(day["pet_mm"], cfg.ai_window_days, cfg.ai_window_min_valid_days)
        lag = pd.Timedelta(days=0 if cfg.ai_window_includes_start else 1)
        for event in ev.itertuples():
            window_end = event.start_date - lag
            in_event = day.loc[event.start_date:event.end_date, "lai_m2m2"]
            lai = {"start_day": day["lai_m2m2"].get(event.start_date),
                   "event_max": in_event.max(), "event_mean": in_event.mean()}[cfg.lai_statistic]
            p, e = p_sum.get(window_end), e_sum.get(window_end)
            rows.append({
                "event_id": event.event_id,
                "precip_window_mm": p, "pet_window_mm": e,
                "ai_60d": p / e if p is not None and e is not None else float("nan"),
                "lai_m2m2": lai,
                "igbp_class": day["igbp_class"].get(event.start_date),
                "tree_cover_pct": day["tree_cover_pct"].get(event.start_date),
            })
    return pd.DataFrame(rows).astype({"ai_60d": float, "lai_m2m2": float,
                                      "precip_window_mm": float, "pet_window_mm": float,
                                      "tree_cover_pct": float})


def event_ranges(event_days: pd.DataFrame) -> pd.DataFrame:
    """Per event: ``event_min_m3m3`` and ``event_max_m3m3`` of the fitted theta."""
    g = event_days.groupby("event_id")["theta_m3m3"]
    return pd.DataFrame({"event_min_m3m3": g.min(), "event_max_m3m3": g.max()}).reset_index()


def attach_covariates(fits_q: pd.DataFrame, events: pd.DataFrame, event_days: pd.DataFrame,
                      sensors: pd.DataFrame, constants: pd.DataFrame,
                      station_day: pd.DataFrame, cfg: Covariates) -> pd.DataFrame:
    """Build the ``analysis`` table (before filters): one row per event with the
    q-model fit, event attributes, sensor constants and covariates.

        frac_range = (event max - event min) / (max_sm - min_sm)        [-]
        mean_depth_m = (depth_top_m + depth_bottom_m) / 2              [m]
    """
    covs = event_covariates(events, station_day, cfg)
    covs["veg_class"] = assign_class(covs["igbp_class"], cfg.class_scheme)
    sensor_cols = ["sensor_id", "station_id", "network", "lat_deg", "lon_deg",
                   "depth_top_m", "depth_bottom_m"]
    out = (fits_q
           .merge(events, on="event_id", how="left", validate="one_to_one")
           .merge(sensors[sensor_cols], on="sensor_id", how="left", validate="many_to_one")
           .merge(constants, on="sensor_id", how="left", validate="many_to_one")
           .merge(event_ranges(event_days), on="event_id", how="left", validate="one_to_one")
           .merge(covs, on="event_id", how="left", validate="one_to_one"))
    return add_derived(out)


def add_derived(analysis: pd.DataFrame) -> pd.DataFrame:
    """Columns derived from others, shared by analyze.py and reproduce.py."""
    analysis = analysis.copy()
    analysis["frac_range"] = ((analysis["event_max_m3m3"] - analysis["event_min_m3m3"])
                              / (analysis["max_sm_m3m3"] - analysis["min_sm_m3m3"]))
    analysis["mean_depth_m"] = (analysis["depth_top_m"] + analysis["depth_bottom_m"]) / 2
    return analysis
