from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SMOKE_PATHS = REPO / "config" / "paths.smoke.yml"


@pytest.fixture
def smoke_cfg():
    from drydown_q.config import load_config
    return load_config("smoke", SMOKE_PATHS)
