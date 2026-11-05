"""Fit loss models to drydown events.

Objective: unweighted least squares in theta over the observed days,

    minimise  sum_i (theta_model(t_i; p) - theta_obs_i)^2   subject to  lb <= p <= ub.

The solver is ``scipy.optimize.least_squares`` (TRF, 2-point Jacobian), called
exactly as ``scipy.optimize.curve_fit`` calls it for a bounded fit, and the
covariance is computed with curve_fit's formula (Moore-Penrose inverse of
J^T J, scaled by SS_res / (n - p)). Calling least_squares directly gives the
optimizer's own active-constraint set, which is reported as ``at_bound_<param>``
without any tolerance. Standard errors are sqrt(diag(covariance)); they are inf
when the covariance is undetermined (n_obs <= number of free parameters).

Parameters (config section ``fitting``):

    q models     free: ``free_params_q`` (default k, q, theta_star); fixed theta_0, theta_w
      theta_0    = min(event max theta, theta_fc)                        (theta_0_rule)
      theta_w    = the sensor's theta_w                                   (soil.theta_w_rule)
      k          in [0, PET_max / (1000 dz)], start k_start_fraction * upper  (k_upper)
                 PET_max = largest daily dPET in the event, pet_default_mm_day if none
      q          in q_bounds, start q_start
      theta_star in theta_star_bounds (named constants), start = upper bound
      A q-model parameter that is not free is fixed at its starting value.
    exponential  free: theta_w in exp_theta_w_bounds (start = their mean), tau in
                 exp_tau_bounds (start exp_tau_start); fixed theta_0 as above.
      derived    k = (theta_fc - theta_w) / tau,  ET_max = 1000 dz k     [mm d-1]

Every fit records ``fit_status`` (``ok``, ``not_converged`` or ``error``) and a
``fit_message``; a failed fit leaves its parameters NaN and never stops the run.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import svd
from scipy.optimize import least_squares

from drydown_q.config import Fitting
from drydown_q.models import MODELS

UNITS = {"theta_0": "theta_0_m3m3", "theta_w": "theta_w_m3m3", "k": "k_per_day", "q": "q",
         "theta_star": "theta_star_m3m3", "tau": "tau_days"}
Q_PARAMS = ["k", "q", "theta_star"]


def _named(bounds: list[str], constants: dict) -> tuple[float, float]:
    """Resolve bounds given as names of sensor/event constants (e.g. min_sm)."""
    return constants[bounds[0]], constants[bounds[1]]


def parameter_setup(model: str, theta_obs: np.ndarray, pet_mm: np.ndarray,
                    sensor: dict, cfg: Fitting) -> tuple[dict, dict]:
    """Return ``(fixed, free)``: fixed values, and ``{name: (lower, upper, start)}``.

    ``sensor`` holds theta_fc_m3m3, min_sm_m3m3, theta_w_m3m3 and dz_m.
    """
    constants = {"min_sm": sensor["min_sm_m3m3"], "theta_fc": sensor["theta_fc_m3m3"],
                 "event_min": np.nanmin(theta_obs), "event_max": np.nanmax(theta_obs)}
    theta_0 = min(constants["event_max"], constants["theta_fc"])
    if model == "exponential":
        w_lo, w_hi = _named(cfg.exp_theta_w_bounds, constants)
        tau_lo, tau_hi = cfg.exp_tau_bounds
        free = {"theta_w": (w_lo, w_hi, np.mean([w_lo, w_hi])),
                "tau": (tau_lo, tau_hi, cfg.exp_tau_start)}
        return {"theta_0": theta_0}, free

    pet_max = np.nanmax(pet_mm) if np.isfinite(pet_mm).any() else cfg.pet_default_mm_day
    k_hi = pet_max / (1000 * sensor["dz_m"])
    star_lo, star_hi = _named(cfg.theta_star_bounds, constants)
    q_lo, q_hi = cfg.q_bounds
    candidates = {"k": (0, k_hi, cfg.k_start_fraction * k_hi),
                  "q": (q_lo, q_hi, cfg.q_start),
                  "theta_star": (star_lo, star_hi, star_hi)}
    fixed = {"theta_0": theta_0, "theta_w": sensor["theta_w_m3m3"]}
    fixed.update({n: candidates[n][2] for n in Q_PARAMS if n not in cfg.free_params_q})
    free = {n: candidates[n] for n in Q_PARAMS if n in cfg.free_params_q}
    return fixed, free


def _call(model: str, t, values: dict):
    f = MODELS[model]
    if model == "exponential":
        return f(t, values["theta_0"], values["theta_w"], values["tau"])
    return f(t, values["theta_0"], values["theta_w"], values["k"], values["q"], values["theta_star"])


def fit_event(t_days: np.ndarray, theta_obs: np.ndarray, pet_mm: np.ndarray,
              sensor: dict, model: str, cfg: Fitting) -> dict:
    """Fit one model to one event; return one ``fits`` row (without event_id).

    ``t_days``/``theta_obs`` are the observed days (NaN theta already dropped);
    ``pet_mm`` is daily dPET over the event days.
    """
    row = {"model": model, "fit_status": "error", "fit_message": "", "r2": np.nan,
           "rmse_m3m3": np.nan}
    try:
        fixed, free = parameter_setup(model, theta_obs, pet_mm, sensor, cfg)
        names = list(free)
        lower = [free[n][0] for n in names]
        upper = [free[n][1] for n in names]
        start = [free[n][2] for n in names]

        def residuals(p):
            return _call(model, t_days, {**fixed, **dict(zip(names, p))}) - theta_obs

        res = least_squares(residuals, start, jac="2-point", bounds=(lower, upper),
                            method="trf", max_nfev=None)
    except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError) as err:
        row["fit_message"] = f"{type(err).__name__}: {err}"
        return row

    values = {**fixed, **dict(zip(names, res.x))}
    for name, value in values.items():
        row[UNITS[name]] = float(value)
    row["fit_status"] = "ok" if res.success else "not_converged"
    row["fit_message"] = res.message
    for name, active in zip(names, res.active_mask):
        row[f"at_bound_{UNITS[name]}"] = bool(active != 0)

    # Covariance exactly as curve_fit computes it for bounded fits.
    n_obs, n_free = len(theta_obs), len(names)
    try:
        _, s, vt = svd(res.jac, full_matrices=False)
        keep = s > np.finfo(float).eps * max(res.jac.shape) * s[0]
        pcov = (vt[keep].T / s[keep] ** 2) @ vt[keep]
        if n_obs > n_free and not np.isnan(pcov).any():
            pcov = pcov * (2 * res.cost) / (n_obs - n_free)
        else:
            pcov = np.full((n_free, n_free), np.inf)
    except (ValueError, np.linalg.LinAlgError) as err:   # e.g. NaN Jacobian
        pcov = np.full((n_free, n_free), np.nan)
        row["fit_message"] += f"; covariance failed: {type(err).__name__}: {err}"
    for name, se in zip(names, np.sqrt(np.diag(pcov))):
        row[f"{UNITS[name]}_se"] = float(se)

    predicted = _call(model, t_days, values)
    resid = theta_obs - predicted
    with np.errstate(divide="ignore", invalid="ignore"):   # one-point events: SS_tot = 0
        row["r2"] = float(1 - np.sum(resid ** 2) / np.sum((theta_obs - theta_obs.mean()) ** 2))
    row["rmse_m3m3"] = float(np.sqrt(np.mean(resid ** 2)))
    if model == "exponential":
        row["k_per_day"] = (sensor["theta_fc_m3m3"] - values["theta_w"]) / values["tau"]
    row["et_max_mm_day"] = 1000 * sensor["dz_m"] * row["k_per_day"]
    return row


def _fit_sensor(events: pd.DataFrame, event_days: pd.DataFrame, sensor: dict,
                cfg: Fitting) -> list[dict]:
    """All models for all events of one sensor (plain loop, no inner parallelism)."""
    rows = []
    days_by_event = dict(tuple(event_days.groupby("event_id", sort=False)))
    for event_id in events["event_id"]:
        days = days_by_event[event_id].sort_values("t_days")
        observed = days[days["theta_m3m3"].notna()]
        for model in cfg.models:
            row = fit_event(observed["t_days"].to_numpy(float), observed["theta_m3m3"].to_numpy(float),
                            days["pet_mm"].to_numpy(float), sensor, model, cfg)
            rows.append({"event_id": event_id, **row})
    return rows


def fit_all(events: pd.DataFrame, event_days: pd.DataFrame, constants: pd.DataFrame,
            cfg: Fitting, n_jobs: int) -> pd.DataFrame:
    """The ``fits`` table: every event x every model in ``cfg.models``.

    Parallel over sensors with joblib (``runtime.n_jobs``); one row per fit
    whatever its outcome.
    """
    sensors = constants.set_index("sensor_id").to_dict("index")
    groups = [(sid, ev, event_days[event_days["event_id"].isin(ev["event_id"])])
              for sid, ev in events.groupby("sensor_id", sort=False)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_sensor)(ev, days, sensors[sid], cfg) for sid, ev, days in groups)
    fits = pd.DataFrame([row for rows in results for row in rows])
    if fits.empty:
        return pd.DataFrame(columns=["event_id", "model", "r2", "rmse_m3m3",
                                     "fit_status", "fit_message"])
    for column in [c for c in fits.columns if c.startswith("at_bound_")]:
        fits[column] = fits[column].astype("boolean")
    for column in ["event_id", "model", "fit_status", "fit_message"]:
        fits[column] = fits[column].astype(str)
    return fits
