"""Guard for the rule "no number with scientific meaning in a .py file".

Every numeric literal in src/ and scripts/ must be one of a few structural
values (indices, counts, formatting, the mm-per-m unit conversion). Anything
else belongs in config/default.yml or data/tables/.
"""

import ast

import pytest

from conftest import REPO

ALLOWED = {
    0, 1, 2, -1,
    1000,      # unit conversion: mm per m (Delta z in m, water depths in mm)
    16,        # length of hex hash prefixes
    20,        # 1 << 20 byte read chunk for checksums
    100,       # unit conversion: percent -> fraction
    10,        # appeears_pipeline station key: first 10 hex characters of SHA-1
    180, 360,  # longitude conventions (degrees)
}
# Matplotlib layout keywords: numbers passed to these are presentation, not science.
LAYOUT_KEYWORDS = {"fontsize", "figsize", "dpi", "ms", "lw", "gridspec_kw", "xytext", "rect",
                   "bbox_to_anchor", "ncol"}
FILES = sorted([*(REPO / "src").rglob("*.py"), *(REPO / "scripts").glob("*.py")])


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_no_numeric_literals_outside_allowlist(path):
    tree = ast.parse(path.read_text())
    layout = {id(n) for kw in ast.walk(tree) if isinstance(kw, ast.keyword) and kw.arg in LAYOUT_KEYWORDS
              for n in ast.walk(kw.value)}
    bad = [(node.lineno, node.value) for node in ast.walk(tree)
           if isinstance(node, ast.Constant) and type(node.value) in (int, float)
           and node.value not in ALLOWED and id(node) not in layout]
    assert not bad, f"numeric literals {bad}: move them to config/default.yml"
