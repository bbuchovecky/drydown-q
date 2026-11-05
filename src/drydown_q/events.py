"""Drydown event detection (analysis step 2), following Morgan's SensorData.

Per sensor, on the daily series theta_t [m3 m-3] (config section ``events``):

0. Masked series. If ``mask_above_fc``, values theta > theta_fc are set
   missing. Missing values (masked or ordinary gaps) are then back-filled with
   the next valid value. A back-filled stretch is flat, so a gap ends an event
   (``gap_rule = end_event``).
1. Thresholds  tau_start = start_min_decline_mm_day / (1000 dz),
               tau_end   = end_min_decline_mm_day   / (1000 dz)   [m3 m-3 d-1]
   with Delta_t = theta_{t+1} - theta_t on the masked series.
2. Day t is a candidate when Delta_t < -tau_start and P_t <= precip_max_mm_day
   (missing P counts as dry when ``precip_missing_is_dry``).
3. A start is the first day of each run of days whose next ``min_decline_days``
   days (the day itself included) are all candidates. Rain is checked only
   there (``precip_check = start_window``).
4. The end is the first day e >= start with Delta_e > -tau_end (Morgan's strict
   comparison). Rain is not checked. If there is none before the record ends,
   end = start (duration 0) and ``end_reason = record_end``, as Morgan did.
   ``end_reason = gap`` when theta is missing on the end day (the event ran
   into a gap), otherwise ``decline_below_threshold``.
5. A start inside an earlier event (start <= that event's end) is skipped:
   events do not overlap. (Morgan's released events contain no overlaps.)

Event values are the unmasked daily theta from start to end; ``event_days``
keeps every day (missing theta included, so PET_max covers all event days)
and the fit drops the missing ones. ``t_days`` counts days since the start.
"""

import numpy as np
import pandas as pd

from drydown_q.config import Events


def masked_series(theta: pd.Series, theta_fc: float, mask_above_fc: bool) -> pd.Series:
    """Step 0: theta with values above theta_fc removed (optional) and back-filled."""
    masked = theta.mask(theta > theta_fc) if mask_above_fc else theta
    return masked.bfill()


def candidate_days(delta: pd.Series, precip: pd.Series, tau_start: float, cfg: Events) -> pd.Series:
    """Step 2: Delta_t < -tau_start and P_t <= precip_max_mm_day."""
    dry = precip <= cfg.precip_max_mm_day
    if cfg.precip_missing_is_dry:
        dry |= precip.isna()
    return (delta < -tau_start) & dry


def find_starts(candidate: pd.Series, min_decline_days: int) -> list[int]:
    """Step 3: positions of the first day of each run of 'the next
    min_decline_days days are all candidates'."""
    c = candidate.to_numpy(bool)
    n, k = len(c), min_decline_days
    if n < k:
        return []
    window = np.convolve(c.astype(int), np.ones(k, dtype=int), mode="valid") == k   # day t covers t..t+k-1
    full = np.zeros(n, dtype=bool)
    full[:len(window)] = window
    return list(np.flatnonzero(full & ~np.concatenate([[False], full[:-1]])))


def detect_sensor(day: pd.DataFrame, theta_fc: float, dz_m: float, cfg: Events) -> pd.DataFrame:
    """Events of one sensor. ``day`` has one row per date (sorted, no gaps in
    the index) with ``date``, ``theta_m3m3``, ``precip_mm``."""
    theta = day["theta_m3m3"].reset_index(drop=True)
    masked = masked_series(theta, theta_fc, cfg.mask_above_fc)
    delta = masked.shift(-1) - masked                                   # Delta_t = theta_{t+1} - theta_t
    tau_start = cfg.start_min_decline_mm_day / (1000 * dz_m)
    tau_end = cfg.end_min_decline_mm_day / (1000 * dz_m)
    candidate = candidate_days(delta, day["precip_mm"].reset_index(drop=True), tau_start, cfg)
    ends_here = (delta > -tau_end).to_numpy()
    dates = day["date"].reset_index(drop=True)
    rows, last_end = [], -1
    for s in find_starts(candidate, cfg.min_decline_days):
        if s <= last_end:
            continue
        after = np.flatnonzero(ends_here[s:])
        if after.size:
            e = s + after[0]
            reason = "gap" if np.isnan(theta[e]) else "decline_below_threshold"
        else:
            e, reason = s, "record_end"
        values = theta[s:e + 1]
        rows.append({"start": s, "end": e, "end_reason": reason,
                     "n_obs": int(values.notna().sum()),
                     "n_days_above_fc": int((values > theta_fc).sum())})
        last_end = e
    out = pd.DataFrame(rows, columns=["start", "end", "end_reason", "n_obs", "n_days_above_fc"])
    out["start_date"] = dates.iloc[out["start"]].to_numpy() if len(out) else pd.Series(dtype="datetime64[ns]")
    out["end_date"] = dates.iloc[out["end"]].to_numpy() if len(out) else pd.Series(dtype="datetime64[ns]")
    return out


def find_events(station_day: pd.DataFrame, constants: pd.DataFrame, cfg: Events
                ) -> dict[str, pd.DataFrame]:
    """The ``events`` and ``event_days`` tables for all sensors."""
    if cfg.precip_check != "start_window" or cfg.precip_source != "chirps" or cfg.exclude_snow_frozen:
        raise ValueError("only precip_check=start_window, precip_source=chirps and "
                         "exclude_snow_frozen=false are implemented so far")
    consts = constants.set_index("sensor_id")
    events, days = [], []
    for sensor_id, day in station_day.groupby("sensor_id", sort=True):
        day = day.sort_values("date")
        day = (day.set_index("date").reindex(pd.date_range(day.date.min(), day.date.max(), freq="D"))
               .rename_axis("date").reset_index())
        c = consts.loc[sensor_id]
        found = detect_sensor(day, c["theta_fc_m3m3"], c["dz_m"], cfg)
        for ev in found.itertuples():
            event_id = f"{sensor_id}/{ev.start_date:%Y-%m-%d}"
            span = day.iloc[ev.start:ev.end + 1]
            events.append({"event_id": event_id, "sensor_id": sensor_id,
                           "start_date": ev.start_date, "end_date": ev.end_date,
                           "duration_days": (ev.end_date - ev.start_date).days,
                           "n_obs": ev.n_obs, "n_days_above_fc": ev.n_days_above_fc,
                           "end_reason": ev.end_reason})
            days.append(pd.DataFrame({"event_id": event_id,
                                      "t_days": np.arange(len(span), dtype="int64"),
                                      "theta_m3m3": span["theta_m3m3"].to_numpy(float),
                                      "pet_mm": span["pet_mm"].to_numpy(float)}))
    events = pd.DataFrame(events, columns=["event_id", "sensor_id", "start_date", "end_date",
                                           "duration_days", "n_obs", "n_days_above_fc", "end_reason"])
    events = events.astype({"event_id": str, "sensor_id": str, "end_reason": str,
                            "start_date": "datetime64[ns]", "end_date": "datetime64[ns]",
                            "duration_days": "int64", "n_obs": "int64", "n_days_above_fc": "int64"})
    event_days = (pd.concat(days, ignore_index=True) if days else
                  pd.DataFrame({"event_id": pd.Series(dtype=str), "t_days": pd.Series(dtype="int64"),
                                "theta_m3m3": pd.Series(dtype=float), "pet_mm": pd.Series(dtype=float)}))
    event_days["event_id"] = event_days["event_id"].astype(str)
    return {"events": events, "event_days": event_days}
