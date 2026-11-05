"""Bin the final event set by AI_60 x LAI within vegetation classes and fit
the q surface to the bin medians.

Surface (``surface_form = "a*ai + b*lai + c*ai*lai + d"``):

    q(AI, LAI) = a * AI + b * LAI + c * AI * LAI + d

    AI   bin midpoint of antecedent aridity AI_60   [-]
    LAI  bin midpoint of leaf area index            [m2 m-2]
    q    median q of the events in the bin          [-]

    dq/dAI = a + c * LAI   (its sign at the class median LAI is a headline result)
"""

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

from drydown_q.config import Binning

COEFFICIENTS = ["a", "b", "c", "d"]


def surface(xy, a, b, c, d):
    """q = a*AI + b*LAI + c*AI*LAI + d (see module docstring)."""
    ai, lai = xy
    return a * ai + b * lai + c * ai * lai + d


def surface_set(analysis: pd.DataFrame, lai_min_m2m2: float, inclusive: bool) -> pd.Series:
    """Events used for bins: the final set, with the surface's own LAI rule
    (``binning.lai_min_inclusive_surface``; Morgan used LAI > 0.5 here while the
    final set used LAI >= 0.5)."""
    lai = analysis["lai_m2m2"]
    lai_ok = lai >= lai_min_m2m2 if inclusive else lai > lai_min_m2m2
    return analysis["in_final_set"] & lai_ok


def midpoints(edges: list[float], open_upper: float) -> np.ndarray:
    """Midpoint of each bin. An infinite last edge is replaced by ``open_upper``."""
    edges = np.asarray(edges, dtype=float)
    if np.isinf(edges[-1]):
        edges[-1] = open_upper
    return (edges[:-1] + edges[1:]) / 2


def bin_medians(events: pd.DataFrame, classes: list[str], cfg: Binning) -> pd.DataFrame:
    """The ``bins`` table: every class x AI bin x LAI bin, including empty ones.

    Bins are left-closed [lo, hi) unless ``bins_right_closed``. Bin codes are
    the position of the bin in ``ai_edges``/``lai_edges``, and midpoints are
    looked up by code, never by order of appearance. ``n_events`` counts events
    with a non-missing q; a bin is ``used_in_fit`` when n_events >=
    ``min_events_per_bin``.
    """
    ev = events[events["veg_class"].isin(classes)].copy()
    ev["ai_bin"] = pd.cut(ev["ai_60d"], cfg.ai_edges, right=cfg.bins_right_closed, labels=False)
    ev["lai_bin"] = pd.cut(ev["lai_m2m2"], cfg.lai_edges, right=cfg.bins_right_closed, labels=False)
    ev = ev.dropna(subset=["ai_bin", "lai_bin", "q"])
    stats = (ev.groupby(["veg_class", "ai_bin", "lai_bin"])["q"]
             .agg(n_events="count", q_median="median").reset_index())

    ai_mid = midpoints(cfg.ai_edges, cfg.open_bin_upper_for_mid["ai"])
    lai_mid = midpoints(cfg.lai_edges, cfg.open_bin_upper_for_mid["lai"])
    grid = pd.MultiIndex.from_product(
        [classes, range(len(ai_mid)), range(len(lai_mid))],
        names=["veg_class", "ai_bin", "lai_bin"]).to_frame(index=False)
    stats["ai_bin"] = stats["ai_bin"].astype(int)
    stats["lai_bin"] = stats["lai_bin"].astype(int)
    bins = grid.merge(stats, on=["veg_class", "ai_bin", "lai_bin"], how="left")
    bins["n_events"] = bins["n_events"].fillna(0).astype("int64")
    bins["ai_mid"] = ai_mid[bins["ai_bin"]]
    bins["lai_mid"] = lai_mid[bins["lai_bin"]]
    bins["used_in_fit"] = (bins["n_events"] >= cfg.min_events_per_bin) & bins["q_median"].notna()
    bins["veg_class"] = bins["veg_class"].astype(str)
    bins["ai_bin"] = bins["ai_bin"].astype("int64")
    bins["lai_bin"] = bins["lai_bin"].astype("int64")
    return bins[["veg_class", "ai_bin", "lai_bin", "ai_mid", "lai_mid",
                 "n_events", "q_median", "used_in_fit"]]


def fit_surface(bins: pd.DataFrame, classes: list[str], cfg: Binning) -> pd.DataFrame:
    """The ``surfaces`` table: least-squares fit of the surface to the used bins
    of each class, every bin weighted equally (``bin_weights = equal``).

    Standard errors are sqrt(diag(covariance)) from ``curve_fit``; ``r2_bins`` is
    1 - SS_res / SS_tot over the used bins. A class with fewer used bins than
    coefficients is recorded with ``fit_status = too_few_bins`` and NaN values.
    """
    rows = []
    for veg_class in classes:
        used = bins[(bins["veg_class"] == veg_class) & bins["used_in_fit"]]
        row = {"veg_class": veg_class, "n_bins": len(used), "n_events": int(used["n_events"].sum())}
        nan = {**{k: np.nan for k in COEFFICIENTS}, **{f"{k}_se": np.nan for k in COEFFICIENTS},
               "r2_bins": np.nan}
        if len(used) < len(COEFFICIENTS):
            rows.append({**row, **nan, "fit_status": "too_few_bins"})
            continue
        xy = (used["ai_mid"].to_numpy(), used["lai_mid"].to_numpy())
        z = used["q_median"].to_numpy()
        try:
            popt, pcov = curve_fit(surface, xy, z)
        except (RuntimeError, OptimizeWarning, ValueError) as err:
            rows.append({**row, **nan, "fit_status": f"failed: {err}"})
            continue
        resid = z - surface(xy, *popt)
        r2 = 1 - np.sum(resid ** 2) / np.sum((z - z.mean()) ** 2)
        se = np.sqrt(np.diag(pcov))
        rows.append({**row, **dict(zip(COEFFICIENTS, popt)),
                     **{f"{k}_se": s for k, s in zip(COEFFICIENTS, se)},
                     "r2_bins": r2, "fit_status": "ok"})
    out = pd.DataFrame(rows)
    out["n_bins"] = out["n_bins"].astype("int64")
    out["n_events"] = out["n_events"].astype("int64")
    return out
