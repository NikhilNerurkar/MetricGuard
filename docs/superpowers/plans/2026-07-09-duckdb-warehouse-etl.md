# DuckDB Warehouse ETL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `etl/load_warehouse.py`, which reads the three raw per-product parquet files in `data/raw/`, standardizes their drifted schemas into one common event schema, and loads them into a DuckDB star schema at `data/warehouse/metricguard.duckdb`, printing a data quality report along the way.

**Architecture:** Medallion-style within one module. Silver stage (pure Python/pandas): three per-product standardization functions normalize column names, timestamp formats, and country-code formats into one common schema; a `pycountry`-backed resolver handles the three different country formats. Gold stage (DuckDB SQL): the unified silver DataFrame is registered as a DuckDB view, and five builder functions create the star schema tables (four dimensions + one fact) with SQL. `main()` orchestrates both stages and prints a data-quality report.

**Tech Stack:** Python 3.12, pandas, duckdb, pycountry, pytest.

## Global Constraints

- Dependencies: add `duckdb` and `pycountry` to `requirements.txt` (alongside existing `pandas`, `numpy`, `pyarrow`, `pytest`).
- No cross-product user identity linking: `dim_users` is keyed by `(source_product, native_user_id)` — the same person on two products gets two separate `dim_users` rows. This is intentional; the raw data has no real overlap signal to resolve.
- `fact_sessions` gets **one row per raw event**, regardless of `event_type` (session_start/session_end/content_view/content_action all included).
- `dim_policy` is **not created** in this task — out of scope.
- Country resolution via `pycountry`: 2-letter input validated as ISO-2; 3-letter input looked up as ISO-3 and converted to ISO-2; anything else resolved by fuzzy name search. Null input stays null. Unresolvable input becomes null (not dropped) and is counted in the data-quality report.
- Messy rows (duplicate ids, nulls, duration outliers) are **loaded as-is** — no filtering or cleaning. Only reported on. Enforcement is a later phase's job.
- `is_qualifying_session = (session_duration_seconds >= 5 AND bot_probability_score < 0.05)`, and is `FALSE` (not null) whenever `session_duration_seconds` is null.
- Output: `data/warehouse/metricguard.duckdb`. The script must be safely re-runnable (`CREATE OR REPLACE TABLE` for every table).
- Data quality report is printed to stdout only — this task does not block or filter on it.

---

### Task 1: Dependencies, module scaffold, and country resolution

**Files:**
- Modify: `requirements.txt`
- Create: `etl/load_warehouse.py`
- Create: `tests/test_load_warehouse.py`

**Interfaces:**
- Produces: `RAW_DIR`, `WAREHOUSE_DIR`, `WAREHOUSE_PATH`, `DURATION_OUTLIER_THRESHOLD_SECONDS = 86_400`, `QUALIFYING_MIN_DURATION_SECONDS = 5`, `QUALIFYING_MAX_BOT_SCORE = 0.05`, `_CONTINENT_MAP: dict[str, tuple[str, str]]`.
- Produces: `_resolve_country_iso2(raw_value) -> str | None`, used by every standardization function in Task 2.

- [ ] **Step 1: Add dependencies**

Add to `requirements.txt` (append to the existing four lines — do not remove them):

```
duckdb
pycountry
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_load_warehouse.py
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'etl.load_warehouse'` (or no attribute `_resolve_country_iso2`).

- [ ] **Step 4: Implement the module scaffold and country resolution**

```python
# etl/load_warehouse.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: `10 passed`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt etl/load_warehouse.py tests/test_load_warehouse.py
git commit -m "Scaffold DuckDB warehouse ETL with country resolution"
```

---

### Task 2: Per-product standardization (silver stage)

**Files:**
- Modify: `etl/load_warehouse.py`
- Modify: `tests/test_load_warehouse.py`

**Interfaces:**
- Consumes: `_resolve_country_iso2` (Task 1).
- Produces: `_standardize_facebook(df)`, `_standardize_instagram(df)`, `_standardize_threads(df)` — each `(pd.DataFrame) -> pd.DataFrame`, returning this exact column order: `source_product, native_event_id, native_user_id, event_type, event_timestamp, raw_country_code, country_iso2, session_duration_seconds, bot_probability_score, product_surface`. `native_event_id` and `native_user_id` are always strings. `event_timestamp` is a proper `datetime64` column. This is the common schema every later task builds on.
- Produces: `load_all_events(raw_dir: Path) -> pd.DataFrame` — reads `facebook.parquet`, `instagram.parquet`, `threads.parquet` from `raw_dir`, standardizes each, and returns one concatenated DataFrame in the schema above.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_load_warehouse.py`:

```python
from etl.load_warehouse import (
    _standardize_facebook, _standardize_instagram, _standardize_threads,
    load_all_events,
)

FACEBOOK_RAW = pd.DataFrame({
    "event_id": ["fb-1", "fb-2"],
    "user_id": [111, 222],
    "event_type": ["session_start", "content_view"],
    "timestamp": ["2026-01-01T10:00:00", "2026-01-02T11:30:00"],
    "country_code": ["US", None],
    "session_duration_seconds": [120.5, None],
    "bot_probability_score": [0.01, 0.2],
    "product_surface": ["Feed", "Groups"],
})

INSTAGRAM_RAW = pd.DataFrame({
    "event_id": [1, 2],
    "uid": ["ig_000111", "ig_000222"],
    "event_type": ["session_start", "content_action"],
    "timestamp": [1767261600, 1767353400],
    "country_code": ["United States", "NGA"],
    "session_duration_seconds": [45.0, 300.0],
    "bot_probability_score": [0.03, 0.5],
    "product_surface": ["Feed", "Reels"],
})

THREADS_RAW = pd.DataFrame({
    "product_surface": ["Feed", "Search"],
    "evt_id": ["th-1", "th-2"],
    "bot_probability_score": [0.02, 0.04],
    "account_id": [333, 444],
    "timestamp": ["2026-01-01T10:00:00", "2026-01-02T11:30:00"],
    "event_type": ["session_start", "content_view"],
    "session_duration_seconds": [60.0, None],
    "country_code": ["USA", "ZAF"],
})

_COMMON_COLUMNS = [
    "source_product", "native_event_id", "native_user_id", "event_type",
    "event_timestamp", "raw_country_code", "country_iso2",
    "session_duration_seconds", "bot_probability_score", "product_surface",
]


def test_standardize_facebook():
    result = _standardize_facebook(FACEBOOK_RAW)
    assert list(result.columns) == _COMMON_COLUMNS
    assert (result["source_product"] == "facebook").all()
    assert result["native_event_id"].tolist() == ["fb-1", "fb-2"]
    assert result["native_user_id"].tolist() == ["111", "222"]
    assert result["event_timestamp"].iloc[0] == pd.Timestamp("2026-01-01T10:00:00")
    assert result["event_timestamp"].iloc[1] == pd.Timestamp("2026-01-02T11:30:00")
    assert result["country_iso2"].iloc[0] == "US"
    assert result["country_iso2"].iloc[1] is None
    assert result["session_duration_seconds"].iloc[0] == 120.5
    assert pd.isna(result["session_duration_seconds"].iloc[1])


def test_standardize_instagram():
    result = _standardize_instagram(INSTAGRAM_RAW)
    assert list(result.columns) == _COMMON_COLUMNS
    assert (result["source_product"] == "instagram").all()
    assert result["native_event_id"].tolist() == ["1", "2"]
    assert result["native_user_id"].tolist() == ["ig_000111", "ig_000222"]
    assert result["event_timestamp"].iloc[0] == pd.Timestamp("2026-01-01T10:00:00")
    assert result["event_timestamp"].iloc[1] == pd.Timestamp("2026-01-02T11:30:00")
    assert result["country_iso2"].tolist() == ["US", "NG"]


def test_standardize_threads():
    result = _standardize_threads(THREADS_RAW)
    assert list(result.columns) == _COMMON_COLUMNS
    assert (result["source_product"] == "threads").all()
    assert result["native_event_id"].tolist() == ["th-1", "th-2"]
    assert result["native_user_id"].tolist() == ["333", "444"]
    assert result["country_iso2"].tolist() == ["US", "ZA"]
    assert result["session_duration_seconds"].iloc[0] == 60.0
    assert pd.isna(result["session_duration_seconds"].iloc[1])


def test_load_all_events(tmp_path):
    FACEBOOK_RAW.to_parquet(tmp_path / "facebook.parquet")
    INSTAGRAM_RAW.to_parquet(tmp_path / "instagram.parquet")
    THREADS_RAW.to_parquet(tmp_path / "threads.parquet")
    result = load_all_events(tmp_path)
    assert len(result) == 6
    assert set(result["source_product"].unique()) == {"facebook", "instagram", "threads"}
    assert list(result.columns) == _COMMON_COLUMNS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: FAIL with `ImportError: cannot import name '_standardize_facebook'`

- [ ] **Step 3: Implement the standardizers and loader**

Append to `etl/load_warehouse.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add etl/load_warehouse.py tests/test_load_warehouse.py
git commit -m "Add per-product standardization and load_all_events"
```

---

### Task 3: Data quality report

**Files:**
- Modify: `etl/load_warehouse.py`
- Modify: `tests/test_load_warehouse.py`

**Interfaces:**
- Consumes: the common event schema from Task 2 (`source_product, native_event_id, ..., country_iso2, raw_country_code, session_duration_seconds, ...`).
- Produces: `print_data_quality_report(all_events: pd.DataFrame) -> None`, printing to stdout. Also produces a test helper `_sample_all_events() -> pd.DataFrame` in the test file, which Tasks 4-6 reuse (do not redefine it in later tasks — import/reuse the one defined here).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_load_warehouse.py`:

```python
from etl.load_warehouse import print_data_quality_report


def _sample_all_events() -> pd.DataFrame:
    return pd.DataFrame({
        "source_product": [
            "facebook", "facebook", "facebook",
            "instagram", "instagram",
            "threads", "threads",
        ],
        "native_event_id": ["fb-1", "fb-2", "fb-2", "ig-1", "ig-2", "th-1", "th-2"],
        "native_user_id": ["111", "222", "222", "555", "666", "333", "444"],
        "event_type": [
            "session_start", "content_view", "content_view",
            "session_start", "content_action",
            "session_start", "content_view",
        ],
        "event_timestamp": pd.to_datetime([
            "2026-01-01T10:00:00", "2026-01-02T11:30:00", "2026-01-02T11:30:00",
            "2026-01-03T09:00:00", "2026-01-03T09:15:00",
            "2026-01-01T10:00:00", "2026-01-04T08:00:00",
        ]),
        "raw_country_code": ["US", None, None, "United States", "NGA", "USA", "Nowhereland"],
        "country_iso2": ["US", None, None, "US", "NG", "US", None],
        "session_duration_seconds": [120.5, None, None, 45.0, 900_000.0, 60.0, -10.0],
        "bot_probability_score": [0.01, 0.2, 0.2, 0.03, 0.5, 0.02, 0.04],
        "product_surface": ["Feed", "Groups", "Groups", "Feed", "Reels", "Feed", "Search"],
    })


def test_print_data_quality_report(capsys):
    print_data_quality_report(_sample_all_events())
    captured = capsys.readouterr().out
    assert "facebook: 3 rows" in captured
    assert "instagram: 2 rows" in captured
    assert "threads: 2 rows" in captured
    assert "fact_sessions (total): 7 rows" in captured
    assert "country_iso2 null rate: 42.86%" in captured
    assert "session_duration_seconds null rate: 28.57%" in captured
    assert "facebook: 1 duplicate native_event_id values" in captured
    assert "instagram: 0 duplicate native_event_id values" in captured
    assert "threads: 0 duplicate native_event_id values" in captured
    assert "facebook: 0 duration outliers" in captured
    assert "instagram: 1 duration outliers" in captured
    assert "threads: 1 duration outliers" in captured
    assert "Unresolved country values (1)" in captured
    assert "Nowhereland" in captured
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: FAIL with `ImportError: cannot import name 'print_data_quality_report'`

- [ ] **Step 3: Implement the data quality report**

Append to `etl/load_warehouse.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add etl/load_warehouse.py tests/test_load_warehouse.py
git commit -m "Add data quality report"
```

---

### Task 4: Gold stage — dim_users and dim_product

**Files:**
- Modify: `etl/load_warehouse.py`
- Modify: `tests/test_load_warehouse.py`

**Interfaces:**
- Consumes: the common event schema (Task 2), `_sample_all_events()` test helper (Task 3 — reuse it, do not redefine).
- Produces: `_build_dim_users(con: duckdb.DuckDBPyConnection) -> None` and `_build_dim_product(con: duckdb.DuckDBPyConnection) -> None`. Both assume a table/view named `all_events` is already registered on `con` (callers' responsibility — Task 6's `main()` registers it). Both create their table via `CREATE OR REPLACE TABLE`. Also produces a test helper `_connection_with_events(all_events) -> duckdb.DuckDBPyConnection` that Tasks 5-6 reuse (do not redefine).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_load_warehouse.py`:

```python
from etl.load_warehouse import _build_dim_users, _build_dim_product


def _connection_with_events(all_events: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.register("all_events", all_events)
    return con


def test_build_dim_users():
    con = _connection_with_events(_sample_all_events())
    _build_dim_users(con)
    result = con.sql("SELECT * FROM dim_users ORDER BY source_product, native_user_id").df()
    assert len(result) == 6
    assert list(result.columns) == [
        "user_id", "source_product", "native_user_id", "account_status", "first_seen_date",
    ]
    assert result["user_id"].nunique() == 6
    assert (result["account_status"] == "active").all()
    fb_222 = result[
        (result["source_product"] == "facebook") & (result["native_user_id"] == "222")
    ].iloc[0]
    assert fb_222["first_seen_date"] == pd.Timestamp("2026-01-02")
    con.close()


def test_build_dim_product():
    con = _connection_with_events(_sample_all_events())
    _build_dim_product(con)
    result = con.sql("SELECT * FROM dim_product ORDER BY product_family, product_surface").df()
    assert len(result) == 6
    assert list(result.columns) == [
        "product_surface_id", "product_family", "product_surface", "display_name",
    ]
    assert (result["display_name"] == result["product_surface"]).all()
    assert result["product_surface_id"].nunique() == 6
    con.close()
```

You will also need `import duckdb` at the top of `tests/test_load_warehouse.py` if it isn't already there (it isn't — earlier tasks didn't need it).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: FAIL with `ImportError: cannot import name '_build_dim_users'`

- [ ] **Step 3: Implement the dimension builders**

Append to `etl/load_warehouse.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: `17 passed`

- [ ] **Step 5: Commit**

```bash
git add etl/load_warehouse.py tests/test_load_warehouse.py
git commit -m "Add dim_users and dim_product builders"
```

---

### Task 5: Gold stage — dim_geography and dim_date

**Files:**
- Modify: `etl/load_warehouse.py`
- Modify: `tests/test_load_warehouse.py`

**Interfaces:**
- Consumes: `_CONTINENT_MAP` (Task 1), `_sample_all_events()` and `_connection_with_events()` test helpers (Tasks 3-4 — reuse, do not redefine).
- Produces: `_build_dim_geography(con) -> None` and `_build_dim_date(con) -> None`, same `CREATE OR REPLACE TABLE` / "assumes `all_events` already registered" contract as Task 4's builders.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_load_warehouse.py`:

```python
from etl.load_warehouse import _build_dim_geography, _build_dim_date


def test_build_dim_geography():
    con = _connection_with_events(_sample_all_events())
    _build_dim_geography(con)
    result = con.sql("SELECT * FROM dim_geography ORDER BY country_id").df()
    assert list(result.columns) == ["country_id", "country_iso2", "country_name", "region", "subregion"]
    assert set(result["country_iso2"].tolist()) == {"US", "NG", None}
    us_row = result[result["country_iso2"] == "US"].iloc[0]
    assert us_row["country_name"] == "United States"
    assert us_row["region"] == "Americas"
    null_row = result[result["country_iso2"].isna()].iloc[0]
    assert null_row["region"] == "Unknown"
    assert null_row["subregion"] == "Unknown"
    con.close()


def test_build_dim_date():
    con = _connection_with_events(_sample_all_events())
    _build_dim_date(con)
    result = con.sql("SELECT * FROM dim_date ORDER BY date").df()
    assert len(result) == 4
    assert list(result.columns) == ["date", "day_of_week", "week", "month", "quarter", "year"]
    assert result["date"].iloc[0] == pd.Timestamp("2026-01-01")
    assert result["date"].iloc[-1] == pd.Timestamp("2026-01-04")
    assert (result["year"] == 2026).all()
    con.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: FAIL with `ImportError: cannot import name '_build_dim_geography'`

- [ ] **Step 3: Implement the geography and date builders**

Append to `etl/load_warehouse.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: `19 passed`

- [ ] **Step 5: Commit**

```bash
git add etl/load_warehouse.py tests/test_load_warehouse.py
git commit -m "Add dim_geography and dim_date builders"
```

---

### Task 6: Gold stage — fact_sessions, orchestration, and real run

**Files:**
- Modify: `etl/load_warehouse.py`
- Modify: `tests/test_load_warehouse.py`

**Interfaces:**
- Consumes: `_build_dim_users`, `_build_dim_product`, `_build_dim_geography`, `_build_dim_date` (Tasks 4-5), `load_all_events`, `print_data_quality_report` (Tasks 2-3), `QUALIFYING_MIN_DURATION_SECONDS`, `QUALIFYING_MAX_BOT_SCORE`, `RAW_DIR`, `WAREHOUSE_PATH` (Task 1). Reuses `_sample_all_events()`, `_connection_with_events()`, `FACEBOOK_RAW`, `INSTAGRAM_RAW`, `THREADS_RAW` test fixtures (Tasks 2-4 — do not redefine).
- Produces: `_build_fact_sessions(con) -> None` (assumes `all_events`, `dim_users`, `dim_product`, `dim_geography` already exist on `con`) and `main(raw_dir: Path = RAW_DIR, warehouse_path: Path = WAREHOUSE_PATH) -> None`, the script's public entry point.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_load_warehouse.py`:

```python
from etl.load_warehouse import (
    _build_fact_sessions, _build_dim_users, _build_dim_product, _build_dim_geography,
    main,
)


def test_build_fact_sessions():
    con = _connection_with_events(_sample_all_events())
    _build_dim_users(con)
    _build_dim_product(con)
    _build_dim_geography(con)
    _build_fact_sessions(con)
    result = con.sql("SELECT * FROM fact_sessions ORDER BY session_id").df()
    assert len(result) == 7
    assert list(result.columns) == [
        "session_id", "user_id", "product_surface_id", "country_id", "event_date",
        "session_duration_seconds", "bot_probability_score", "is_qualifying_session",
        "event_type", "native_event_id", "source_product",
    ]
    assert result["user_id"].notna().all()
    assert result["product_surface_id"].notna().all()
    assert result["country_id"].notna().all()
    assert int(result["is_qualifying_session"].sum()) == 3
    con.close()


def test_main_builds_warehouse(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    FACEBOOK_RAW.to_parquet(raw_dir / "facebook.parquet")
    INSTAGRAM_RAW.to_parquet(raw_dir / "instagram.parquet")
    THREADS_RAW.to_parquet(raw_dir / "threads.parquet")
    warehouse_path = tmp_path / "warehouse" / "metricguard.duckdb"

    main(raw_dir=raw_dir, warehouse_path=warehouse_path)

    assert warehouse_path.exists()
    con = duckdb.connect(str(warehouse_path))
    tables = {row[0] for row in con.sql("SHOW TABLES").fetchall()}
    assert tables == {"dim_users", "dim_product", "dim_geography", "dim_date", "fact_sessions"}
    fact_count = con.sql("SELECT COUNT(*) FROM fact_sessions").fetchone()[0]
    assert fact_count == 6
    con.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: FAIL with `ImportError: cannot import name '_build_fact_sessions'`

- [ ] **Step 3: Implement fact_sessions and main()**

Append to `etl/load_warehouse.py`:

```python
def _build_fact_sessions(con: duckdb.DuckDBPyConnection) -> None:
    con.sql(f"""
        CREATE OR REPLACE TABLE fact_sessions AS
        SELECT
            ROW_NUMBER() OVER () AS session_id,
            u.user_id,
            p.product_surface_id,
            g.country_id,
            CAST(e.event_timestamp AS DATE) AS event_date,
            e.session_duration_seconds,
            e.bot_probability_score,
            COALESCE(
                e.session_duration_seconds >= {QUALIFYING_MIN_DURATION_SECONDS}
                AND e.bot_probability_score < {QUALIFYING_MAX_BOT_SCORE},
                FALSE
            ) AS is_qualifying_session,
            e.event_type,
            e.native_event_id,
            e.source_product
        FROM all_events e
        JOIN dim_users u
            ON u.source_product = e.source_product AND u.native_user_id = e.native_user_id
        JOIN dim_product p
            ON p.product_family = e.source_product AND p.product_surface = e.product_surface
        LEFT JOIN dim_geography g
            ON g.country_iso2 IS NOT DISTINCT FROM e.country_iso2
    """)


def main(raw_dir: Path = RAW_DIR, warehouse_path: Path = WAREHOUSE_PATH) -> None:
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    all_events = load_all_events(raw_dir)
    print_data_quality_report(all_events)

    con = duckdb.connect(str(warehouse_path))
    con.register("all_events", all_events)
    _build_dim_users(con)
    _build_dim_product(con)
    _build_dim_geography(con)
    _build_dim_date(con)
    _build_fact_sessions(con)
    con.close()
    print(f"Warehouse written to {warehouse_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_load_warehouse.py -v`
Expected: `21 passed`

- [ ] **Step 5: Run the script for real and inspect output**

Run: `python etl/load_warehouse.py`
Expected: the data quality report prints to stdout, followed by `Warehouse written to .../data/warehouse/metricguard.duckdb`, and that file exists on disk. Spot-check it:

```bash
python -c "
import duckdb
con = duckdb.connect('data/warehouse/metricguard.duckdb')
print(con.sql('SELECT COUNT(*) FROM fact_sessions').fetchone())
print(con.sql('SELECT * FROM dim_geography ORDER BY country_id').df())
con.close()
"
```

Expected: `fact_sessions` row count matches the total row count across the three raw parquet files (~150,000), and `dim_geography` shows resolved country names/regions with no crash.

- [ ] **Step 6: Commit**

```bash
git add etl/load_warehouse.py tests/test_load_warehouse.py data/warehouse/metricguard.duckdb
git commit -m "Add fact_sessions builder and main() orchestration"
```
