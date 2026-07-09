# DuckDB Star Schema ETL — Design

## Purpose

Load the three raw per-product parquet files in `data/raw/` (produced by
`etl/generate_synthetic_data.py`) into a governed DuckDB star schema at
`data/warehouse/metricguard.duckdb`, resolving the intentional cross-product
schema drift (different id/user column names, timestamp encodings, and
country-code formats) into one consistent fact/dimension model that later
phases (semantic layer, governed views, validation gate) can build on.

Implemented in `etl/load_warehouse.py`.

## Architecture

Medallion-style, three stages inside one script:

- **Bronze:** the existing `data/raw/*.parquet` files, untouched.
- **Silver (Python):** one standardization function per product, each
  reading its raw parquet and returning a DataFrame in one common schema.
  Row-wise messiness — timestamp format differences and country-format
  resolution — is handled here in Python because it's awkward to express
  as set-based SQL.
- **Gold (DuckDB SQL):** the unified silver DataFrame is registered as a
  DuckDB view/table, and the star schema (dimensions + fact) is built from
  it with SQL — surrogate keys, joins, and aggregation-ready structure are
  DuckDB's strength.

## Silver stage: per-product standardization

Each of `_standardize_facebook`, `_standardize_instagram`,
`_standardize_threads` reads one raw parquet file and returns a DataFrame
with this common schema:

| column | type | notes |
|---|---|---|
| `source_product` | str | `"facebook"` / `"instagram"` / `"threads"` |
| `native_event_id` | str | the product's original id column, cast to string (Instagram's is int64) |
| `native_user_id` | str | the product's original user column, cast to string |
| `event_type` | str | unchanged |
| `event_timestamp` | datetime64 | parsed from ISO string (Facebook/Threads) or Unix epoch int (Instagram) |
| `country_iso2` | str or null | resolved via pycountry (see below); null in, null or unresolved out |
| `session_duration_seconds` | float | unchanged, including negative/outlier/null values — no clipping |
| `bot_probability_score` | float | unchanged |
| `product_surface` | str | unchanged |

The three standardized frames are concatenated into one `all_events`
DataFrame (~150K rows). No rows are dropped at this stage.

### Country resolution

`_resolve_country_iso2(raw_value) -> str | None`, using the `pycountry`
library:
- Already 2 letters → validate via `pycountry.countries.get(alpha_2=...)`.
- 3 letters → look up via `pycountry.countries.get(alpha_3=...)`, return its
  `alpha_2`.
- Otherwise → treat as a full name, resolve via
  `pycountry.countries.search_fuzzy(...)`, return the first match's
  `alpha_2`.
- Null input → null output.
- No match found (any of the above raises/returns nothing) → null output,
  and the original raw value is counted as "unresolved" in the data quality
  report (not dropped from the data).

## Gold stage: DuckDB star schema

Written to `data/warehouse/metricguard.duckdb`. All tables are rebuilt with
`CREATE OR REPLACE TABLE` so the script is safely re-runnable.

- **`dim_users`**: surrogate `user_id` (int) per distinct
  `(source_product, native_user_id)` pair — no cross-product identity
  linking (the source data has no real overlap signal to resolve).
  `account_status` is a constant `'active'` for every row (the raw data has
  no status field; this is a known synthetic-data limitation, documented in
  the script, not something the ETL invents data to fill). `first_seen_date`
  = min `event_date` for that user.
- **`dim_product`**: surrogate `product_surface_id` per distinct
  `(source_product, product_surface)`. `product_family` = `source_product`.
  `display_name` = `product_surface` (human-readable as-is).
- **`dim_geography`**: surrogate `country_id` per distinct resolved
  `country_iso2` (nulls get their own row, `country_iso2 = NULL`).
  `country_name` via `pycountry.countries.get(alpha_2=...).name`. `region`
  and `subregion` come from a small hardcoded ISO2→continent dict (covering
  the ~15 countries the generator produces); anything not in the dict gets
  `'Unknown'` for both — no new dependency for this.
- **`dim_date`**: one row per calendar date spanning
  `min(event_date)..max(event_date)` actually present in the data (not a
  fixed calendar-year skeleton), with `day_of_week`, `week`, `month`,
  `quarter`, `year` computed via DuckDB date functions.
- **`dim_policy`**: **not created.** Nothing in the raw data or
  `fact_sessions` references policy/violations yet; this belongs to a later
  phase (content-action policy enforcement).
- **`fact_sessions`**: one row per raw event (all four `event_type` values
  included, not just session_start/session_end — matches how the generator
  actually produced the data). Columns:
  - `session_id` (surrogate, since native ids can collide due to injected
    duplicate-id messiness — they cannot be the PK)
  - `user_id` (FK → dim_users)
  - `product_surface_id` (FK → dim_product)
  - `country_id` (FK → dim_geography, nullable when country unresolved)
  - `event_date` (FK → dim_date)
  - `session_duration_seconds`, `bot_probability_score` (unchanged from silver)
  - `is_qualifying_session` (bool: `duration_seconds >= 5 AND bot_probability_score < 0.05`;
    false when duration is null)
  - `event_type`, `native_event_id`, `source_product` — kept for
    traceability/debugging even though the ROADMAP's table listing doesn't
    mention them; an explicit, called-out addition rather than a silent one.

## Data quality report

Printed to stdout at the end of the run (not enforced/blocking — enforcement
is Phase 6's job, not this script's):
- Row counts: each raw source file, and final `fact_sessions`.
- Null rate for `country_iso2` (post-resolution) and `session_duration_seconds`.
- Duplicate `native_event_id` count, per product.
- Duration-outlier count (negative or > 86,400 seconds / 24h), per product.
- Distinct list (capped) of raw country values that failed pycountry
  resolution.

## File structure

Single file, `etl/load_warehouse.py`, matching the existing repo convention
(one file per ETL script, as in `etl/generate_synthetic_data.py`). Internally
organized as clearly separated functions: three standardizers, the country
resolver, five dimension/fact builder functions, the data-quality report
function, and `main()`.

## Dependencies

`duckdb`, `pycountry` — added to `requirements.txt` (alongside the existing
`pandas`, `numpy`, `pyarrow`, `pytest`).

## Testing

Same approach as the Phase 1 generator: unit tests per standardization/
builder function against small fixture DataFrames (not the full ~150K-row
files, for speed), plus one end-to-end test that runs `main()` against tiny
fixtures pointed at a temp directory and checks the resulting `.duckdb` file
contains the expected tables with the expected row counts.

## Out of scope

- `dim_policy` and any policy/violation modeling (later phase).
- Cross-product user identity resolution (no real signal in the data to
  resolve; revisit only if the Phase 1 generator is later changed to inject
  genuine overlapping users).
- Filtering/cleaning messy rows (duplicates, nulls, outliers) — they are
  loaded as-is; only reported on. Enforcement belongs to the Phase 6
  validation gate.
- `semantic_layer/`, `validation/`, `agent/` — untouched by this task.
