"""MODIS station extractions (LAI, MCD12Q1 land cover, MOD44B tree cover).

Inputs are the tidy parquet files of the ``appeears_pipeline`` package in
``paths.appeears_dir``: ``<group>__<product>.parquet`` with columns ``sid``,
``di``, ``dj``, ``date`` and the layers. ``sid`` is that package's station key,
the first 10 hex characters of SHA-1("network/station"); ``di = dj = 0`` is the
station pixel (the other offsets are a 3 x 3 neighbourhood, not used here).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from drydown_q.config import TABLES_DIR, Prepare

SID_LENGTH = 10   # appeears_pipeline.build.station_id: sha1(...)[:10]


def appeears_sid(station_id: str) -> str:
    """appeears_pipeline's key for ``network/station``."""
    return hashlib.sha1(station_id.encode()).hexdigest()[:SID_LENGTH]


def product_file(appeears_dir: Path, product: str) -> Path:
    matches = sorted(Path(appeears_dir).glob(f"*__{product}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one *__{product}.parquet in {appeears_dir}, found {matches}")
    return matches[0]


def _station_pixel(table: pd.DataFrame, station_ids: pd.Series) -> pd.DataFrame:
    key = pd.DataFrame({"station_id": station_ids.unique()})
    key["sid"] = key["station_id"].map(appeears_sid)
    centre = table[(table["di"] == 0) & (table["dj"] == 0)]
    return centre.merge(key, on="sid", how="inner")


def daily_lai(lai: pd.DataFrame, station_ids: pd.Series, cfg: Prepare) -> pd.DataFrame:
    """Daily LAI [m2 m-2] per station: ``station_id, date, lai_m2m2``.

    ``lai_qc = none`` (Morgan's Station.get_lai): keep the delivered values
    below ``lai_keep_below`` (MODIS codes 248-255 are fill/class codes), with no
    QC bits; place each 8-day composite on its start date and forward-fill
    daily, for at most ``lai_fill_max_days`` days (null = unlimited). Uses
    ``Lai_500m_raw`` (as delivered by AppEEARS) when present.
    """
    if cfg.lai_qc != "none":
        raise ValueError(f"prepare.lai_qc = {cfg.lai_qc!r} is not implemented yet")
    px = _station_pixel(lai, station_ids)
    raw = px["Lai_500m_raw"] if "Lai_500m_raw" in px else px["Lai_500m"]
    px = px.assign(lai=raw.where(raw < cfg.lai_keep_below), date=pd.to_datetime(px["date"]))
    out = []
    for station_id, s in px.groupby("station_id"):
        series = s.set_index("date")["lai"].sort_index()
        series = series[~series.index.duplicated()]
        daily = series.resample("D").asfreq().ffill(limit=cfg.lai_fill_max_days)
        out.append(pd.DataFrame({"station_id": station_id, "date": daily.index,
                                 "lai_m2m2": daily.to_numpy(float)}))
    if not out:
        return pd.DataFrame(columns=["station_id", "date", "lai_m2m2"])
    return pd.concat(out, ignore_index=True)


def landcover_by_year(lc: pd.DataFrame, vcf: pd.DataFrame, station_ids: pd.Series) -> pd.DataFrame:
    """Per station and calendar year: ``igbp_class`` (MCD12Q1 ``LC_Type1`` mapped
    through data/tables/igbp_classes.csv) and ``tree_cover_pct`` (MOD44B
    ``Percent_Tree_Cover``; values outside the product's valid range are already
    NaN in the tidy files)."""
    igbp = pd.read_csv(TABLES_DIR / "igbp_classes.csv").set_index("igbp_number")["igbp_class"]
    lc_px = _station_pixel(lc, station_ids)
    lc_px = pd.DataFrame({"station_id": lc_px["station_id"],
                          "year": pd.to_datetime(lc_px["date"]).dt.year,
                          "igbp_class": lc_px["LC_Type1"].map(igbp)})
    vcf_px = _station_pixel(vcf, station_ids)
    vcf_px = pd.DataFrame({"station_id": vcf_px["station_id"],
                           "year": pd.to_datetime(vcf_px["date"]).dt.year,
                           "tree_cover_pct": vcf_px["Percent_Tree_Cover"].astype(float)})
    out = lc_px.merge(vcf_px, on=["station_id", "year"], how="outer")
    out["igbp_class"] = out["igbp_class"].astype(object).where(out["igbp_class"].notna(), np.nan)
    return out
