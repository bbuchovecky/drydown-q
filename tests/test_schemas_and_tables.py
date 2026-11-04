"""Committed data: the smoke dataset follows the table schemas, and the lookup
tables in data/tables are complete."""

import pandas as pd
import pytest

from drydown_q.config import TABLES_DIR
from drydown_q.schemas import assert_schema
from conftest import REPO

SMOKE = REPO / "tests" / "data" / "smoke"


@pytest.mark.parametrize("table", ["sensors", "station_day"])
def test_smoke_tables_follow_schema(table):
    assert_schema(pd.read_parquet(SMOKE / f"{table}.parquet"), table)


def test_smoke_dataset_has_three_stations_one_per_class():
    sensors = pd.read_parquet(SMOKE / "sensors.parquet")
    day = pd.read_parquet(SMOKE / "station_day.parquet")
    assert sensors["station_id"].nunique() == 3
    assert set(day["igbp_class"]) == {"GRA", "SAV", "ENF"}
    assert set(day["sensor_id"]) == set(sensors["sensor_id"])


def test_unexpected_column_and_wrong_dtype_are_rejected():
    sensors = pd.read_parquet(SMOKE / "sensors.parquet")
    with pytest.raises(AssertionError, match="unexpected column"):
        assert_schema(sensors.assign(extra=1.0), "sensors")
    with pytest.raises(AssertionError, match="expected float"):
        assert_schema(sensors.assign(lat_deg="north"), "sensors")


def test_class_scheme_covers_every_igbp_class():
    igbp = pd.read_csv(TABLES_DIR / "igbp_classes.csv")
    schemes = pd.read_csv(TABLES_DIR / "class_schemes.csv")
    for scheme, rows in schemes.groupby("scheme"):
        assert set(rows["igbp_class"]) == set(igbp["igbp_class"]), scheme
        assert rows["igbp_class"].is_unique, scheme
    morgan = schemes[schemes.scheme == "morgan_3"].set_index("igbp_class")["veg_class"]
    assert morgan["GRA"] == "HERB" and morgan["SAV"] == "SAV" and morgan["OSH"] == "SHR"
    assert {morgan[c] for c in ["WSA", "ENF", "EBF", "DBF", "MF"]} == {"WOOD"}


def test_clapp_hornberger_table_has_eleven_classes():
    ch = pd.read_csv(TABLES_DIR / "clapp_hornberger_1978.csv")
    assert len(ch) == 11 and ch["texture_class"].is_unique
    rules = pd.read_csv(TABLES_DIR / "usda_texture_rules.csv")
    # "silt" has a texture rule but no C&H parameters: those sensors fall back to max_sm.
    assert set(rules["texture_class"]) - set(ch["texture_class"]) == {"silt"}
