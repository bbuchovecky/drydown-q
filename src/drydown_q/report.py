"""Run records: the manifest (provenance) and, later, the human-readable report."""

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from drydown_q.config import Config, as_dict

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
