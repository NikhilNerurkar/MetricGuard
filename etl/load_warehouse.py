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


def _standardize_facebook(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "source_product": "facebook",
        "native_event_id": df["event_id"].astype(str),
        "native_user_id": df["user_id"].astype(str),
        "event_type": df["event_type"],
        "event_timestamp": pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S"),
        "raw_country_code": df["country_code"],
        "country_iso2": df["country_code"].apply(_resolve_country_iso2),
        "session_duration_seconds": df["session_duration_seconds"],
        "bot_probability_score": df["bot_probability_score"],
        "product_surface": df["product_surface"],
    })


def _standardize_instagram(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "source_product": "instagram",
        "native_event_id": df["event_id"].astype(str),
        "native_user_id": df["uid"].astype(str),
        "event_type": df["event_type"],
        "event_timestamp": pd.to_datetime(df["timestamp"], unit="s"),
        "raw_country_code": df["country_code"],
        "country_iso2": df["country_code"].apply(_resolve_country_iso2),
        "session_duration_seconds": df["session_duration_seconds"],
        "bot_probability_score": df["bot_probability_score"],
        "product_surface": df["product_surface"],
    })


def _standardize_threads(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "source_product": "threads",
        "native_event_id": df["evt_id"].astype(str),
        "native_user_id": df["account_id"].astype(str),
        "event_type": df["event_type"],
        "event_timestamp": pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S"),
        "raw_country_code": df["country_code"],
        "country_iso2": df["country_code"].apply(_resolve_country_iso2),
        "session_duration_seconds": df["session_duration_seconds"],
        "bot_probability_score": df["bot_probability_score"],
        "product_surface": df["product_surface"],
    })


def load_all_events(raw_dir: Path) -> pd.DataFrame:
    facebook = _standardize_facebook(pd.read_parquet(raw_dir / "facebook.parquet"))
    instagram = _standardize_instagram(pd.read_parquet(raw_dir / "instagram.parquet"))
    threads = _standardize_threads(pd.read_parquet(raw_dir / "threads.parquet"))
    return pd.concat([facebook, instagram, threads], ignore_index=True)
