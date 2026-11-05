"""Run records: the manifest (provenance) and, later, the human-readable report."""

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from drydown_q.config import Config, as_dict
from drydown_q.schemas import assert_schema

PACKAGES = ["numpy", "pandas", "scipy", "xarray", "netCDF4", "pyarrow", "matplotlib",
            "pyyaml", "joblib", "ismn", "drydown-q"]


def file_checksum(path: Path) -> str:
    """SHA-256 of a file's bytes (inputs are recorded in the manifest by this)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def write_manifest(run_dir: Path, cfg: Config, code: str, inputs: dict[str, str],
                   stage_keys: dict[str, str], started: float) -> Path:
    """Write ``manifest.json``: resolved config, code hash, input checksums,
    cache key of every stage, package versions and wall time."""
    manifest = {
        "run": cfg.run_name,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1),
        "code_hash": code,
        "inputs_sha256": inputs,
        "stage_keys": stage_keys,
        "python": platform.python_version(),
        "packages": package_versions(),
        "config": as_dict(cfg),
    }
    path = Path(run_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def markdown_table(df, float_format: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub markdown table (no extra dependency)."""
    def cell(v):
        if isinstance(v, float):
            if v != v:
                return ""
            return str(int(v)) if v.is_integer() else float_format.format(v)
        return str(v)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def write_report(path: Path, title: str, sections: list[tuple[str, str]]) -> Path:
    """Write ``report.md``: a title, then each (heading, markdown body) in order."""
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines += [f"## {heading}", "", body, ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def compare_events(ours: pd.DataFrame, released: pd.DataFrame,
                   parameters: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match our events with released ones on (sensor_id, start_date).

    ``ours`` and ``released`` each have ``sensor_id``, ``start_date`` and the
    parameter columns (``parameters`` maps our column -> released column).
    Returns ``(summary, matched)``: the summary has the share of released events
    reproduced, the share of our events that are new, the same shares restricted
    to sensors present in both, and the median |difference| of each parameter
    over matched events; ``matched`` is the joined table.
    """
    key = ["sensor_id", "start_date"]
    rel = released[key + list(parameters.values())].rename(columns={v: f"released_{k}" for k, v in parameters.items()})
    matched = ours[key + list(parameters)].merge(rel, on=key, how="inner")
    common = set(ours["sensor_id"]) & set(released["sensor_id"])
    rows = [
        ("released events", len(released)), ("our events", len(ours)), ("matched events", len(matched)),
        ("share of released events reproduced", len(matched) / max(len(released), 1)),
        ("share of our events that are new", 1 - len(matched) / max(len(ours), 1)),
        ("sensors in both", len(common)),
        ("share of released sensors present in ours", len(common) / max(released["sensor_id"].nunique(), 1)),
        ("share reproduced, sensors in both",
         len(matched) / max(released["sensor_id"].isin(common).sum(), 1)),
        ("share new, sensors in both", 1 - len(matched) / max(ours["sensor_id"].isin(common).sum(), 1)),
    ]
    for ours_col in parameters:
        diff = (matched[ours_col] - matched[f"released_{ours_col}"]).abs()
        rows.append((f"median |difference| {ours_col}", diff.median()))
    return pd.DataFrame(rows, columns=["quantity", "value"]), matched


def match_sensors(ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    """Pair our sensors with another sensor list (e.g. Morgan's metadata.csv).

    Both tables have network, station, sensor, depth_top_m, depth_bottom_m.
    Stations are compared case-insensitively (ISMN changed the case of station
    names between archive versions, e.g. AGUACANYON -> AguaCanyon). Sensors are
    paired first by exact name, then by (network, station, depth range) when
    that key is unique on both sides (instrument names also change). Returns
    ``our_sensor_id``, ``network``, ``their_station``, ``their_sensor``, ``match``
    in {name, depth}. Replicate probes at one depth stay unpaired here.
    """
    o = ours.assign(station_key=ours["station"].str.lower())
    t = theirs.assign(station_key=theirs["station"].str.lower()).rename(
        columns={"station": "their_station", "sensor": "their_sensor"})
    name = o.merge(t[["network", "station_key", "their_station", "their_sensor"]],
                   left_on=["network", "station_key", "sensor"],
                   right_on=["network", "station_key", "their_sensor"]).assign(match="name")
    rest_o = o[~o["sensor_id"].isin(name["sensor_id"])]
    taken = name.set_index(["network", "station_key", "their_sensor"]).index
    rest_t = t[~t.set_index(["network", "station_key", "their_sensor"]).index.isin(taken)]
    dkey = ["network", "station_key", "depth_top_m", "depth_bottom_m"]
    uniq_o = rest_o[~rest_o.duplicated(dkey, keep=False)]
    uniq_t = rest_t[~rest_t.duplicated(dkey, keep=False)]
    depth = uniq_o.merge(uniq_t[dkey + ["their_station", "their_sensor"]], on=dkey).assign(match="depth")
    out = pd.concat([name, depth], ignore_index=True)
    return out[["sensor_id", "network", "their_station", "their_sensor", "match"]].rename(
        columns={"sensor_id": "our_sensor_id"})


# ── Sweep summaries (sweep.py, reproduce.py --sweep) ──────────────────────────
SURFACE_FIELDS = ["a", "b", "c", "d", "r2_bins", "n_bins", "dq_dai_at_median_lai"]


def sweep_row(label: str, overrides: dict, status: str, message: str, result: dict | None,
              classes: list[str]) -> dict:
    """One ``summary`` row. ``result`` has events_final, stations_final,
    events_surface, median_q {class: q} and surfaces {class: {a, b, ...}}."""
    changed = {f"{s}.{k}": v for s, kv in overrides.items() if s != "runtime" for k, v in kv.items()}
    row = {"run": label, "changed_key": ", ".join(changed) or "(baseline)",
           "changed_value": ", ".join(json.dumps(v) for v in changed.values()),
           "status": status, "message": message,
           "events_final": np.nan, "stations_final": np.nan, "events_surface": np.nan}
    if result is not None:
        row.update({k: result[k] for k in ("events_final", "stations_final", "events_surface")})
        for c in classes:
            row[f"median_q_{c}"] = result["median_q"].get(c, np.nan)
            surf = result["surfaces"].get(c, {})
            for k in SURFACE_FIELDS:
                row[f"{k}_{c}"] = surf.get(k, np.nan)
            # Headline: the sign of dq/dAI = a + c * LAI at the class median LAI.
            row[f"dq_dai_sign_{c}"] = np.sign(row[f"dq_dai_at_median_lai_{c}"])
    return row


def sweep_summary(rows: list[dict]) -> pd.DataFrame:
    """The ``summary`` table, with the headline grassland - woodland contrast."""
    summary = pd.DataFrame(rows)
    if {"median_q_HERB", "median_q_WOOD"} <= set(summary.columns):
        summary["herb_minus_wood_q"] = summary["median_q_HERB"] - summary["median_q_WOOD"]
    for column in summary.columns:
        if column not in ("run", "changed_key", "changed_value", "status", "message"):
            summary[column] = summary[column].astype(float)
    assert_schema(summary, "summary")
    return summary


def specification_curve(summary: pd.DataFrame, classes: list[str], path: Path, title: str) -> Path:
    """Each headline result against run rank, with the varied setting of each
    run as a dot matrix underneath. Runs are ranked by the HERB - WOOD median-q
    difference (all panels share that order); the baseline is a dashed line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = summary[summary["status"] == "ok"].copy()
    order = "herb_minus_wood_q" if "herb_minus_wood_q" in ok else f"median_q_{classes[0]}"
    ok = ok.sort_values(order, ignore_index=True)
    x = np.arange(len(ok))
    panels = [("median q", [f"median_q_{c}" for c in classes]), ("HERB − WOOD q", ["herb_minus_wood_q"]),
              *[(f"surface {k}", [f"{k}_{c}" for c in classes]) for k in ["a", "b", "c", "d"]],
              ("R² on bins", [f"r2_bins_{c}" for c in classes]),
              ("sign ∂q/∂AI at median LAI", [f"dq_dai_sign_{c}" for c in classes])]
    settings = [k for k in dict.fromkeys(ok["changed_key"]) if k != "(baseline)"]
    fig, axes = plt.subplots(len(panels) + 1, 1, sharex=True,
                             figsize=(max(6, 0.35 * len(ok) + 2), 2 * len(panels) + 3),
                             gridspec_kw={"height_ratios": [1] * len(panels) + [max(1, 0.25 * len(settings))]})
    base_x = x[ok["run"] == "baseline"]
    colours = dict(zip(classes, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for ax, (panel_title, columns) in zip(axes, panels):
        for column in columns:
            if column in ok:
                veg_class = column.rsplit("_", 1)[-1]
                ax.plot(x, ok[column], "o", ms=4, color=colours.get(veg_class, "k"), label=veg_class)
        for bx in base_x:
            ax.axvline(bx, color="0.6", lw=0.8, ls="--")
        ax.set_ylabel(panel_title, fontsize=8)
    axes[0].legend(fontsize=7, ncol=len(classes), loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False)
    matrix = axes[-1]
    for i, row in ok.iterrows():
        if row["changed_key"] in settings:
            matrix.plot(i, settings.index(row["changed_key"]), "ks", ms=5)
            matrix.annotate(row["changed_value"], (i, settings.index(row["changed_key"])), fontsize=6,
                            xytext=(0, 5), textcoords="offset points", ha="center")
    matrix.set_yticks(range(len(settings)), settings, fontsize=7)
    matrix.set_ylim(-1, len(settings))
    matrix.set_xticks(x)
    matrix.set_xlabel(f"run rank (by {order}); dashed line = baseline", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)
