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


def print_data_quality_report(all_events: pd.DataFrame) -> None:
    print("=== Data Quality Report ===")
    for product in ["facebook", "instagram", "threads"]:
        count = int((all_events["source_product"] == product).sum())
        print(f"{product}: {count} rows")
    print(f"fact_sessions (total): {len(all_events)} rows")

    country_null_rate = all_events["country_iso2"].isna().mean()
    duration_null_rate = all_events["session_duration_seconds"].isna().mean()
    print(f"country_iso2 null rate: {country_null_rate:.2%}")
    print(f"session_duration_seconds null rate: {duration_null_rate:.2%}")

    for product in ["facebook", "instagram", "threads"]:
        subset = all_events[all_events["source_product"] == product]
        dup_count = int(subset["native_event_id"].duplicated().sum())
        print(f"{product}: {dup_count} duplicate native_event_id values")

    duration = all_events["session_duration_seconds"]
    outlier_mask = (duration < 0) | (duration > DURATION_OUTLIER_THRESHOLD_SECONDS)
    for product in ["facebook", "instagram", "threads"]:
        product_mask = all_events["source_product"] == product
        count = int((outlier_mask & product_mask).sum())
        print(f"{product}: {count} duration outliers (negative or > 24h)")

    unresolved_mask = all_events["country_iso2"].isna() & all_events["raw_country_code"].notna()
    unresolved = sorted(all_events.loc[unresolved_mask, "raw_country_code"].unique().tolist())
    print(f"Unresolved country values ({len(unresolved)}): {unresolved[:20]}")


def _build_dim_users(con: duckdb.DuckDBPyConnection) -> None:
    con.sql("""
        CREATE OR REPLACE TABLE dim_users AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY source_product, native_user_id) AS user_id,
            source_product,
            native_user_id,
            'active' AS account_status,
            MIN(CAST(event_timestamp AS DATE)) AS first_seen_date
        FROM all_events
        GROUP BY source_product, native_user_id
    """)


def _build_dim_product(con: duckdb.DuckDBPyConnection) -> None:
    con.sql("""
        CREATE OR REPLACE TABLE dim_product AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY source_product, product_surface) AS product_surface_id,
            source_product AS product_family,
            product_surface,
            product_surface AS display_name
        FROM (SELECT DISTINCT source_product, product_surface FROM all_events)
    """)


def _build_dim_geography(con: duckdb.DuckDBPyConnection) -> None:
    distinct_countries = con.sql(
        "SELECT DISTINCT country_iso2 FROM all_events"
    ).df()["country_iso2"].tolist()

    rows = []
    for iso2 in distinct_countries:
        if iso2 is None:
            rows.append({
                "country_iso2": None,
                "country_name": None,
                "region": "Unknown",
                "subregion": "Unknown",
            })
            continue
        match = pycountry.countries.get(alpha_2=iso2)
        country_name = match.name if match else None
        region, subregion = _CONTINENT_MAP.get(iso2, ("Unknown", "Unknown"))
        rows.append({
            "country_iso2": iso2,
            "country_name": country_name,
            "region": region,
            "subregion": subregion,
        })

    geography_df = pd.DataFrame(rows, columns=["country_iso2", "country_name", "region", "subregion"])
    geography_df.insert(0, "country_id", range(1, len(geography_df) + 1))

    con.register("_dim_geography_stage", geography_df)
    con.sql("CREATE OR REPLACE TABLE dim_geography AS SELECT * FROM _dim_geography_stage")


def _build_dim_date(con: duckdb.DuckDBPyConnection) -> None:
    con.sql("""
        CREATE OR REPLACE TABLE dim_date AS
        SELECT
            d::DATE AS date,
            DAYNAME(d) AS day_of_week,
            WEEK(d) AS week,
            MONTH(d) AS month,
            QUARTER(d) AS quarter,
            YEAR(d) AS year
        FROM (
            SELECT UNNEST(GENERATE_SERIES(
                (SELECT MIN(CAST(event_timestamp AS DATE)) FROM all_events),
                (SELECT MAX(CAST(event_timestamp AS DATE)) FROM all_events),
                INTERVAL 1 DAY
            )) AS d
        )
    """)
