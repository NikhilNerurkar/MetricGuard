# Synthetic Event Data Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `etl/generate_synthetic_data.py`, a script that generates ~50,000 rows of synthetic event-log data each for Facebook, Instagram, and Threads, with intentional per-product schema drift and injected data-quality problems, and writes them to `data/raw/*.parquet`.

**Architecture:** A single module with four pure, independently testable functions layered on top of each other: `_generate_canonical_events` (shared row generation) → `_apply_schema_drift` (per-product renaming/typing/reordering) → `_inject_messiness` (per-product row corruption) → `generate_product_events` (orchestrates the three for one product) → `main` (loops over all three products and writes Parquet files). Each layer is a pure function taking and returning a `pandas.DataFrame`, driven by a single shared `numpy.random.Generator` so the whole run is reproducible from one seed.

**Tech Stack:** Python 3.12, pandas, numpy, pyarrow (Parquet I/O), pytest.

## Global Constraints

- Reproducibility: seed a single `numpy.random.Generator` with `SEED = 42`; running `main()` twice must produce byte-identical DataFrames.
- Row count: 49,500–50,500 rows per product (randomized per product, not a fixed exact count).
- Timestamp window: uniform-random within the last 90 days from "now" at run time (`LOOKBACK_DAYS = 90`).
- Messiness: ~1–3% of rows per product get duplicate ids, nulls in country/duration, duration outliers, and out-of-order timestamps — each injected independently.
- Output files: `data/raw/facebook.parquet`, `data/raw/instagram.parquet`, `data/raw/threads.parquet`.
- Dependencies: add `pandas`, `numpy`, `pyarrow`, `pytest` to `requirements.txt` (currently empty).
- Out of scope: `etl/load_warehouse.py`, `etl/build_governed_views.py`, and anything under `validation/` — this plan only produces the raw Parquet files.

---

### Task 1: Project scaffolding — dependencies, import path, product config

**Files:**
- Create: `etl/generate_synthetic_data.py`
- Create: `conftest.py` (repo root)
- Create: `tests/test_generate_synthetic_data.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `PRODUCT_CONFIGS: dict[str, dict]` with keys `"facebook"`, `"instagram"`, `"threads"`, each a dict with keys `id_col`, `id_style`, `user_col`, `user_style`, `timestamp_style`, `country_style`, `duration_style`, `surfaces`, `column_order`.
- Produces: `SEED = 42`, `ROW_COUNT_RANGE = (49_500, 50_500)`, `LOOKBACK_DAYS = 90`, `EVENT_TYPES`, `EVENT_TYPE_WEIGHTS`, `COUNTRIES` (used by Task 2 onward).

- [ ] **Step 1: Add dependencies**

Write `requirements.txt`:

```
pandas
numpy
pyarrow
pytest
```

- [ ] **Step 2: Add root conftest.py so `import etl...` resolves under pytest**

```python
# conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

- [ ] **Step 3: Write the failing test for product config shape**

```python
# tests/test_generate_synthetic_data.py
from etl.generate_synthetic_data import PRODUCT_CONFIGS


def test_product_configs_defines_all_three_products():
    assert set(PRODUCT_CONFIGS.keys()) == {"facebook", "instagram", "threads"}


def test_product_configs_have_required_keys():
    required_keys = {
        "id_col", "id_style", "user_col", "user_style",
        "timestamp_style", "country_style", "duration_style",
        "surfaces", "column_order",
    }
    for product, config in PRODUCT_CONFIGS.items():
        assert required_keys.issubset(config.keys())
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'etl.generate_synthetic_data'` (or the module has no attribute `PRODUCT_CONFIGS`).

- [ ] **Step 5: Implement constants and product config**

```python
# etl/generate_synthetic_data.py
"""Synthetic event log generator for Facebook, Instagram, and Threads."""
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROW_COUNT_RANGE = (49_500, 50_500)
LOOKBACK_DAYS = 90

EVENT_TYPES = ["session_start", "session_end", "content_view", "content_action"]
EVENT_TYPE_WEIGHTS = [0.20, 0.20, 0.45, 0.15]

# Each entry: (iso2, iso3, full_name)
COUNTRIES = [
    ("US", "USA", "United States"),
    ("GB", "GBR", "United Kingdom"),
    ("CA", "CAN", "Canada"),
    ("DE", "DEU", "Germany"),
    ("FR", "FRA", "France"),
    ("IN", "IND", "India"),
    ("BR", "BRA", "Brazil"),
    ("AU", "AUS", "Australia"),
    ("JP", "JPN", "Japan"),
    ("MX", "MEX", "Mexico"),
    ("ES", "ESP", "Spain"),
    ("NG", "NGA", "Nigeria"),
    ("ZA", "ZAF", "South Africa"),
    ("KR", "KOR", "South Korea"),
    ("SG", "SGP", "Singapore"),
]

PRODUCT_CONFIGS = {
    "facebook": {
        "id_col": "event_id",
        "id_style": "uuid",
        "user_col": "user_id",
        "user_style": "int",
        "timestamp_style": "iso",
        "country_style": "iso2",
        "duration_style": "float",
        "surfaces": ["Feed", "Groups", "Marketplace", "Video"],
        "column_order": None,
    },
    "instagram": {
        "id_col": "event_id",
        "id_style": "int_sequential",
        "user_col": "uid",
        "user_style": "ig_string",
        "timestamp_style": "epoch",
        "country_style": "mixed",
        "duration_style": "int",
        "surfaces": ["Feed", "Reels", "Stories", "Explore"],
        "column_order": None,
    },
    "threads": {
        "id_col": "evt_id",
        "id_style": "uuid",
        "user_col": "account_id",
        "user_style": "int",
        "timestamp_style": "iso",
        "country_style": "iso3",
        "duration_style": "float_null_nonsession",
        "surfaces": ["Feed", "Profile", "Search"],
        "column_order": [
            "product_surface", "evt_id", "bot_probability_score",
            "account_id", "timestamp", "event_type",
            "session_duration_seconds", "country_code",
        ],
    },
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: `2 passed`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt conftest.py etl/generate_synthetic_data.py tests/test_generate_synthetic_data.py
git commit -m "Scaffold synthetic data generator with product config"
```

---

### Task 2: Canonical event generator (shared row generation)

**Files:**
- Modify: `etl/generate_synthetic_data.py`
- Modify: `tests/test_generate_synthetic_data.py`

**Interfaces:**
- Consumes: `EVENT_TYPES`, `EVENT_TYPE_WEIGHTS`, `COUNTRIES`, `LOOKBACK_DAYS` from Task 1.
- Produces: `_generate_canonical_events(n: int, surfaces: list[str], rng: np.random.Generator, now: datetime) -> pd.DataFrame` with columns `event_id` (str uuid), `user_id` (int), `event_type` (str), `timestamp` (`datetime`), `country_code` (tuple `(iso2, iso3, name)`), `session_duration_seconds` (float), `bot_probability_score` (float), `product_surface` (str). This column set and order is the "canonical order" later tasks reorder from.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generate_synthetic_data.py`:

```python
from datetime import datetime, timedelta, timezone

import pandas as pd

from etl.generate_synthetic_data import (
    _generate_canonical_events,
    EVENT_TYPES,
    EVENT_TYPE_WEIGHTS,
    LOOKBACK_DAYS,
)

FIXED_NOW = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


def test_canonical_events_row_count():
    rng = np.random.default_rng(1)
    df = _generate_canonical_events(1000, ["Feed"], rng, FIXED_NOW)
    assert len(df) == 1000


def test_canonical_events_event_type_proportions():
    rng = np.random.default_rng(1)
    df = _generate_canonical_events(50_000, ["Feed"], rng, FIXED_NOW)
    counts = df["event_type"].value_counts(normalize=True)
    for event_type, expected in zip(EVENT_TYPES, EVENT_TYPE_WEIGHTS):
        assert abs(counts[event_type] - expected) < 0.03


def test_canonical_events_bot_score_in_range():
    rng = np.random.default_rng(1)
    df = _generate_canonical_events(1000, ["Feed"], rng, FIXED_NOW)
    assert df["bot_probability_score"].between(0, 1).all()


def test_canonical_events_timestamps_within_lookback_window():
    rng = np.random.default_rng(1)
    df = _generate_canonical_events(1000, ["Feed"], rng, FIXED_NOW)
    earliest = FIXED_NOW - timedelta(days=LOOKBACK_DAYS)
    assert (df["timestamp"] >= earliest).all()
    assert (df["timestamp"] <= FIXED_NOW).all()


def test_canonical_events_reproducible():
    df1 = _generate_canonical_events(1000, ["Feed"], np.random.default_rng(7), FIXED_NOW)
    df2 = _generate_canonical_events(1000, ["Feed"], np.random.default_rng(7), FIXED_NOW)
    pd.testing.assert_frame_equal(df1, df2)
```

Also add `import numpy as np` at the top of the test file if not already present from Task 1 (it is not — Task 1's test only imported `PRODUCT_CONFIGS`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: FAIL with `ImportError: cannot import name '_generate_canonical_events'`

- [ ] **Step 3: Implement `_generate_canonical_events`**

Append to `etl/generate_synthetic_data.py`:

```python
def _generate_canonical_events(
    n: int, surfaces: list[str], rng: np.random.Generator, now: datetime
) -> pd.DataFrame:
    event_types = rng.choice(EVENT_TYPES, size=n, p=EVENT_TYPE_WEIGHTS)

    offsets_seconds = rng.uniform(0, LOOKBACK_DAYS * 86400, size=n)
    timestamps = [now - timedelta(seconds=float(s)) for s in offsets_seconds]

    country_idx = rng.integers(0, len(COUNTRIES), size=n)
    countries = [COUNTRIES[i] for i in country_idx]

    durations = rng.lognormal(mean=4.5, sigma=1.0, size=n)
    bot_scores = rng.beta(2, 8, size=n)
    surface_choices = rng.choice(surfaces, size=n)
    user_ids = rng.integers(100_000, 999_999, size=n)

    raw_bytes = rng.bytes(16 * n)
    event_ids = [
        str(uuid.UUID(bytes=raw_bytes[i * 16 : (i + 1) * 16])) for i in range(n)
    ]

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "user_id": user_ids,
            "event_type": event_types,
            "timestamp": timestamps,
            "country_code": countries,
            "session_duration_seconds": durations,
            "bot_probability_score": bot_scores,
            "product_surface": surface_choices,
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add etl/generate_synthetic_data.py tests/test_generate_synthetic_data.py
git commit -m "Add canonical event generator"
```

---

### Task 3: Per-product schema drift

**Files:**
- Modify: `etl/generate_synthetic_data.py`
- Modify: `tests/test_generate_synthetic_data.py`

**Interfaces:**
- Consumes: output of `_generate_canonical_events` (Task 2), `PRODUCT_CONFIGS` (Task 1).
- Produces: `_apply_schema_drift(df: pd.DataFrame, product: str, config: dict, rng: np.random.Generator) -> pd.DataFrame` — renames/casts/reorders columns per the product's config. Column names after this step are the final, per-product names Task 4 and Task 5 operate on (e.g. `evt_id`/`account_id` for Threads, `uid` for Instagram).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generate_synthetic_data.py`:

```python
from etl.generate_synthetic_data import _apply_schema_drift, PRODUCT_CONFIGS


def _make_canonical_df(n=200, surfaces=("Feed",), seed=3):
    rng = np.random.default_rng(seed)
    return _generate_canonical_events(n, list(surfaces), rng, FIXED_NOW), rng


def test_facebook_schema_drift():
    df, rng = _make_canonical_df()
    config = PRODUCT_CONFIGS["facebook"]
    drifted = _apply_schema_drift(df, "facebook", config, rng)
    assert list(drifted.columns) == [
        "event_id", "user_id", "event_type", "timestamp",
        "country_code", "session_duration_seconds",
        "bot_probability_score", "product_surface",
    ]
    assert drifted["event_id"].apply(lambda v: isinstance(v, str)).all()
    assert drifted["timestamp"].str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$").all()
    assert drifted["country_code"].apply(len).eq(2).all()


def test_instagram_schema_drift():
    df, rng = _make_canonical_df()
    config = PRODUCT_CONFIGS["instagram"]
    drifted = _apply_schema_drift(df, "instagram", config, rng)
    assert list(drifted.columns) == [
        "event_id", "uid", "event_type", "timestamp",
        "country_code", "session_duration_seconds",
        "bot_probability_score", "product_surface",
    ]
    assert pd.api.types.is_integer_dtype(drifted["event_id"])
    assert list(drifted["event_id"]) == list(range(1, len(drifted) + 1))
    assert drifted["uid"].str.match(r"^ig_\d{6}$").all()
    assert pd.api.types.is_integer_dtype(drifted["timestamp"])
    assert pd.api.types.is_integer_dtype(drifted["session_duration_seconds"])
    name_lengths = drifted["country_code"].apply(len)
    assert name_lengths.gt(3).any()  # some full country names present
    assert name_lengths.eq(2).any()  # some ISO-2 codes present


def test_threads_schema_drift():
    df, rng = _make_canonical_df()
    config = PRODUCT_CONFIGS["threads"]
    drifted = _apply_schema_drift(df, "threads", config, rng)
    assert list(drifted.columns) == config["column_order"]
    assert drifted["evt_id"].apply(lambda v: isinstance(v, str)).all()
    assert drifted["country_code"].apply(len).eq(3).all()
    non_session_mask = ~drifted["event_type"].isin(["session_start", "session_end"])
    assert drifted.loc[non_session_mask, "session_duration_seconds"].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: FAIL with `ImportError: cannot import name '_apply_schema_drift'`

- [ ] **Step 3: Implement `_apply_schema_drift`**

Append to `etl/generate_synthetic_data.py`:

```python
_CANONICAL_DURATION_COL = "session_duration_seconds"


def _apply_schema_drift(
    df: pd.DataFrame, product: str, config: dict, rng: np.random.Generator
) -> pd.DataFrame:
    df = df.copy()

    if config["id_style"] == "int_sequential":
        df["event_id"] = np.arange(1, len(df) + 1, dtype=np.int64)
    if config["id_col"] != "event_id":
        df = df.rename(columns={"event_id": config["id_col"]})

    if config["user_style"] == "ig_string":
        df["user_id"] = df["user_id"].apply(lambda uid: f"ig_{uid % 1_000_000:06d}")
    if config["user_col"] != "user_id":
        df = df.rename(columns={"user_id": config["user_col"]})

    if config["timestamp_style"] == "epoch":
        df["timestamp"] = df["timestamp"].apply(lambda ts: int(ts.timestamp()))
    else:
        df["timestamp"] = df["timestamp"].apply(
            lambda ts: ts.strftime("%Y-%m-%dT%H:%M:%S")
        )

    if config["country_style"] == "iso2":
        df["country_code"] = df["country_code"].apply(lambda c: c[0])
    elif config["country_style"] == "iso3":
        df["country_code"] = df["country_code"].apply(lambda c: c[1])
    elif config["country_style"] == "mixed":
        use_iso2 = rng.random(len(df)) < 0.5
        df["country_code"] = [
            c[0] if flag else c[2] for c, flag in zip(df["country_code"], use_iso2)
        ]

    if config["duration_style"] == "int":
        df[_CANONICAL_DURATION_COL] = (
            df[_CANONICAL_DURATION_COL].round().astype("int64")
        )
    elif config["duration_style"] == "float_null_nonsession":
        is_session = df["event_type"].isin(["session_start", "session_end"])
        df.loc[~is_session, _CANONICAL_DURATION_COL] = np.nan

    canonical_order = [
        config["id_col"], config["user_col"], "event_type", "timestamp",
        "country_code", _CANONICAL_DURATION_COL,
        "bot_probability_score", "product_surface",
    ]
    order = config["column_order"] or canonical_order
    return df[order]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add etl/generate_synthetic_data.py tests/test_generate_synthetic_data.py
git commit -m "Add per-product schema drift"
```

---

### Task 4: Messiness injection

**Files:**
- Modify: `etl/generate_synthetic_data.py`
- Modify: `tests/test_generate_synthetic_data.py`

**Interfaces:**
- Consumes: output of `_apply_schema_drift` (Task 3), `PRODUCT_CONFIGS` (Task 1).
- Produces: `_inject_messiness(df: pd.DataFrame, config: dict, rng: np.random.Generator) -> pd.DataFrame` — same columns/dtypes as input, ~1-3% of rows corrupted (duplicate ids, null country/duration, duration outliers, out-of-order timestamps).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generate_synthetic_data.py`:

```python
from etl.generate_synthetic_data import _inject_messiness


def _make_drifted_df(product, n=1000, seed=5):
    rng = np.random.default_rng(seed)
    config = PRODUCT_CONFIGS[product]
    canonical = _generate_canonical_events(n, config["surfaces"], rng, FIXED_NOW)
    drifted = _apply_schema_drift(canonical, product, config, rng)
    return drifted, config, rng


def test_messiness_injects_duplicate_ids():
    df, config, rng = _make_drifted_df("facebook")
    messy = _inject_messiness(df, config, rng)
    assert messy[config["id_col"]].duplicated().sum() >= 1


def test_messiness_injects_nulls():
    df, config, rng = _make_drifted_df("facebook")
    messy = _inject_messiness(df, config, rng)
    assert messy["country_code"].isna().sum() >= 1
    assert messy["session_duration_seconds"].isna().sum() >= 1


def test_messiness_injects_duration_outliers():
    df, config, rng = _make_drifted_df("facebook")
    messy = _inject_messiness(df, config, rng)
    durations = messy["session_duration_seconds"].dropna()
    assert (durations < 0).any() or (durations > 500_000).any()


def test_messiness_injects_out_of_order_timestamps_iso_product():
    df, config, rng = _make_drifted_df("facebook")
    messy = _inject_messiness(df, config, rng)
    parsed = pd.to_datetime(messy["timestamp"])
    earliest_allowed = FIXED_NOW - timedelta(days=LOOKBACK_DAYS)
    assert (parsed < earliest_allowed).any()


def test_messiness_injects_out_of_order_timestamps_epoch_product():
    df, config, rng = _make_drifted_df("instagram")
    messy = _inject_messiness(df, config, rng)
    earliest_allowed_epoch = int((FIXED_NOW - timedelta(days=LOOKBACK_DAYS)).timestamp())
    assert (messy["timestamp"] < earliest_allowed_epoch).any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: FAIL with `ImportError: cannot import name '_inject_messiness'`

- [ ] **Step 3: Implement `_inject_messiness`**

Append to `etl/generate_synthetic_data.py`:

```python
def _inject_messiness(
    df: pd.DataFrame, config: dict, rng: np.random.Generator
) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    n = len(df)
    id_col = config["id_col"]
    country_col = "country_code"
    duration_col = _CANONICAL_DURATION_COL
    timestamp_col = "timestamp"

    dup_count = max(1, int(n * 0.01))
    dup_targets = rng.choice(n, size=dup_count, replace=False)
    dup_sources = rng.choice(n, size=dup_count, replace=False)
    df.loc[dup_targets, id_col] = df.loc[dup_sources, id_col].values

    null_count = max(1, int(n * 0.02))
    df.loc[rng.choice(n, size=null_count, replace=False), country_col] = np.nan
    df.loc[rng.choice(n, size=null_count, replace=False), duration_col] = np.nan

    outlier_count = max(1, int(n * 0.01))
    outlier_idx = rng.choice(n, size=outlier_count, replace=False)
    outlier_values = rng.choice([-1, -60, 750_000, 900_000], size=outlier_count)
    df[duration_col] = df[duration_col].astype("float64")
    df.loc[outlier_idx, duration_col] = outlier_values.astype("float64")

    reorder_count = max(1, int(n * 0.01))
    reorder_idx = rng.choice(n, size=reorder_count, replace=False)
    shift_days = rng.integers(7, 30, size=reorder_count)
    if config["timestamp_style"] == "epoch":
        df[timestamp_col] = df[timestamp_col].astype("int64")
        df.loc[reorder_idx, timestamp_col] = (
            df.loc[reorder_idx, timestamp_col] - shift_days * 86400
        )
    else:
        parsed = pd.to_datetime(df.loc[reorder_idx, timestamp_col])
        shifted = parsed - pd.to_timedelta(shift_days, unit="D")
        df.loc[reorder_idx, timestamp_col] = shifted.dt.strftime("%Y-%m-%dT%H:%M:%S")

    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add etl/generate_synthetic_data.py tests/test_generate_synthetic_data.py
git commit -m "Add messiness injection"
```

---

### Task 5: Orchestration, main(), and Parquet output

**Files:**
- Modify: `etl/generate_synthetic_data.py`
- Modify: `tests/test_generate_synthetic_data.py`

**Interfaces:**
- Consumes: `_generate_canonical_events` (Task 2), `_apply_schema_drift` (Task 3), `_inject_messiness` (Task 4), `PRODUCT_CONFIGS`, `SEED`, `ROW_COUNT_RANGE`, `OUTPUT_DIR` (Task 1).
- Produces: `generate_product_events(product: str, rng: np.random.Generator, now: datetime | None = None) -> pd.DataFrame` and `main(output_dir: Path = OUTPUT_DIR, now: datetime | None = None) -> None`, which writes `<output_dir>/facebook.parquet`, `<output_dir>/instagram.parquet`, `<output_dir>/threads.parquet`. `main` accepts `now` (forwarded to every `generate_product_events` call) so tests can pin the clock — production calls leave it `None` and each product generation call uses the real current time independently.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generate_synthetic_data.py`:

```python
from etl.generate_synthetic_data import (
    generate_product_events, main, ROW_COUNT_RANGE, SEED,
)


def test_generate_product_events_row_count_in_range():
    rng = np.random.default_rng(SEED)
    df = generate_product_events("facebook", rng, now=FIXED_NOW)
    assert ROW_COUNT_RANGE[0] <= len(df) <= ROW_COUNT_RANGE[1]


def test_main_writes_three_parquet_files(tmp_path):
    main(output_dir=tmp_path)
    for product in PRODUCT_CONFIGS:
        path = tmp_path / f"{product}.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert ROW_COUNT_RANGE[0] <= len(df) <= ROW_COUNT_RANGE[1]


def test_main_is_reproducible(tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    main(output_dir=out1, now=FIXED_NOW)
    main(output_dir=out2, now=FIXED_NOW)
    for product in PRODUCT_CONFIGS:
        df1 = pd.read_parquet(out1 / f"{product}.parquet")
        df2 = pd.read_parquet(out2 / f"{product}.parquet")
        pd.testing.assert_frame_equal(df1, df2)
```

`FIXED_NOW` pins the clock so both runs format identical timestamp strings — with the real clock, two back-to-back runs could straddle a second boundary and produce different second-precision timestamp strings for the same row, making this assertion flaky.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_product_events'`

- [ ] **Step 3: Implement `generate_product_events` and `main`**

Append to `etl/generate_synthetic_data.py`:

```python
def generate_product_events(
    product: str, rng: np.random.Generator, now: datetime | None = None
) -> pd.DataFrame:
    config = PRODUCT_CONFIGS[product]
    now = now or datetime.now(timezone.utc)
    n = int(rng.integers(ROW_COUNT_RANGE[0], ROW_COUNT_RANGE[1] + 1))
    canonical = _generate_canonical_events(n, config["surfaces"], rng, now)
    drifted = _apply_schema_drift(canonical, product, config, rng)
    return _inject_messiness(drifted, config, rng)


def main(output_dir: Path = OUTPUT_DIR, now: datetime | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for product in PRODUCT_CONFIGS:
        df = generate_product_events(product, rng, now=now)
        path = output_dir / f"{product}.parquet"
        df.to_parquet(path, engine="pyarrow", index=False)
        print(f"{product}: wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_generate_synthetic_data.py -v`
Expected: `18 passed`

- [ ] **Step 5: Run the script for real and inspect output**

Run: `python etl/generate_synthetic_data.py`
Expected: three lines like `facebook: wrote 498xx rows to .../data/raw/facebook.parquet`, and `data/raw/facebook.parquet`, `data/raw/instagram.parquet`, `data/raw/threads.parquet` exist on disk.

- [ ] **Step 6: Commit**

```bash
git add etl/generate_synthetic_data.py tests/test_generate_synthetic_data.py data/raw/*.parquet
git commit -m "Add orchestration, main(), and Parquet output"
```
