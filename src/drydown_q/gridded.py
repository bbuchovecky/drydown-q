"""Gridded daily forcing (CHIRPS precipitation, dPET) at stations.

``extract_at_stations`` is added in build phase 4. ``window_sum`` is used both by
prepare.py (stored 60-day sums) and by the covariates step (configured window).
"""

import pandas as pd


def window_sum(daily: pd.Series, window_days: int, min_valid_days: int) -> pd.Series:
    """Trailing sum over the ``window_days`` calendar days ending on (and
    including) each date.

        S(t) = sum_{d = t - window_days + 1}^{t} x(d)      [same unit as x]

    ``daily`` is indexed by date (one row per day; missing rows count as
    missing days). S(t) is NaN when fewer than ``min_valid_days`` of the window
    are non-missing.
    """
    daily = daily.sort_index()
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    sums = full.rolling(window_days, min_periods=min_valid_days).sum()
    return sums.reindex(daily.index)
