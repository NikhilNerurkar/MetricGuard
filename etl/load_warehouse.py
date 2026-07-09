"""ETL pipeline that loads standardized event data into a DuckDB star schema."""
from pathlib import Path

import duckdb
import pandas as pd
import pycountry

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
WAREHOUSE_DIR = Path(__file__).resolve().parent.parent / "data" / "warehouse"
WAREHOUSE_PATH = WAREHOUSE_DIR / "metricguard.duckdb"

DURATION_OUTLIER_THRESHOLD_SECONDS = 86_400
QUALIFYING_MIN_DURATION_SECONDS = 5
QUALIFYING_MAX_BOT_SCORE = 0.05

# ISO2 -> (region, subregion), covering the countries the synthetic
# generator produces. Anything outside this map falls back to "Unknown".
_CONTINENT_MAP: dict[str, tuple[str, str]] = {
    "US": ("Americas", "Northern America"),
    "GB": ("Europe", "Northern Europe"),
    "CA": ("Americas", "Northern America"),
    "DE": ("Europe", "Western Europe"),
    "FR": ("Europe", "Western Europe"),
    "IN": ("Asia", "Southern Asia"),
    "BR": ("Americas", "South America"),
    "AU": ("Oceania", "Australia and New Zealand"),
    "JP": ("Asia", "Eastern Asia"),
    "MX": ("Americas", "Central America"),
    "ES": ("Europe", "Southern Europe"),
    "NG": ("Africa", "Western Africa"),
    "ZA": ("Africa", "Southern Africa"),
    "KR": ("Asia", "Eastern Asia"),
    "SG": ("Asia", "South-Eastern Asia"),
}


def _resolve_country_iso2(raw_value) -> str | None:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return None
    value = str(raw_value).strip()
    if len(value) == 2:
        match = pycountry.countries.get(alpha_2=value.upper())
        return match.alpha_2 if match else None
    if len(value) == 3:
        match = pycountry.countries.get(alpha_3=value.upper())
        return match.alpha_2 if match else None
    try:
        matches = pycountry.countries.search_fuzzy(value)
    except LookupError:
        return None
    return matches[0].alpha_2 if matches else None
