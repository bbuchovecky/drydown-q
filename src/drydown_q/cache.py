"""Content-addressed cache for pipeline stages.

A stage's outputs live in ``<cache_dir>/<stage>/<key>/<table>.parquet``. The key
hashes the stage's config section, the upstream stage's key and the code hash,
so a stage is recomputed exactly when one of those changes. A ``_complete.json``
marker is written last; a folder without it (an interrupted run) is recomputed.
"""

import json
import time
from pathlib import Path

import pandas as pd

from drydown_q.config import config_hash
from drydown_q.schemas import assert_schema


def cached_stage(stage: str, compute, cache_dir: Path, section, upstream: str,
                 code: str) -> tuple[dict[str, pd.DataFrame], str]:
    """Return ``(tables, key)`` for ``stage``, computing it only if not cached.

    ``compute()`` must return ``{table_name: DataFrame}``; every table is checked
    against its schema in ``schemas.py`` before it is written.
    """
    key = config_hash(stage, section, upstream, code)
    folder = Path(cache_dir) / stage / key
    marker = folder / "_complete.json"
    if marker.exists():
        names = json.loads(marker.read_text())["tables"]
        return {name: pd.read_parquet(folder / f"{name}.parquet") for name in names}, key

    started = time.time()
    tables = compute()
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        assert_schema(table, name)
        tmp = folder / f"{name}.parquet.part"
        table.to_parquet(tmp, index=False)
        tmp.rename(folder / f"{name}.parquet")
    marker.write_text(json.dumps({
        "stage": stage, "tables": list(tables), "upstream": upstream, "code": code,
        "rows": {name: len(t) for name, t in tables.items()},
        "seconds": round(time.time() - started, 2),
    }, indent=2))
    return tables, key
