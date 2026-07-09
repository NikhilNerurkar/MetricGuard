import pandas as pd
import pytest

from etl.load_warehouse import _resolve_country_iso2


@pytest.mark.parametrize("raw_value,expected", [
    ("US", "US"),
    ("us", "US"),
    ("USA", "US"),
    ("United States", "US"),
    ("Nigeria", "NG"),
    ("NGA", "NG"),
    ("South Korea", "KR"),
    (None, None),
])
def test_resolve_country_iso2_known_values(raw_value, expected):
    assert _resolve_country_iso2(raw_value) == expected


def test_resolve_country_iso2_nan_input():
    assert _resolve_country_iso2(float("nan")) is None


def test_resolve_country_iso2_unresolvable_value():
    assert _resolve_country_iso2("Nowhereland") is None
