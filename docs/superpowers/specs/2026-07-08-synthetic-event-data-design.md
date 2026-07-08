# Synthetic Event Data Generator — Design

## Purpose

Seed `data/raw/` with per-product event-log data for three products — Facebook,
Instagram, and Threads — that the rest of the MetricGuard pipeline (warehouse
load → governed views → semantic layer → validation/lineage/gate) can be
built and tested against. The schemas are intentionally heterogeneous across
products, and each product's data carries a small amount of injected
messiness, so the downstream validation/governance layer has real,
representative problems to catch.

Implemented in `etl/generate_synthetic_data.py`.

## Shared mechanics

- Reproducible: `random.seed(42)` and `np.random.seed(42)` set at the top of
  the script. Same output on every run.
- Timestamp range: uniform-random within the last 90 days, i.e.
  `[today - 90 days, today]` evaluated at run time (not hard-coded dates).
- Row count: ~50,000 rows per product, randomized in the 49,500–50,500 range
  per product rather than an exact fixed count.
- `event_type` proportions (shared across products): `session_start` ~20%,
  `session_end` ~20%, `content_view` ~45%, `content_action` ~15%.
- `bot_probability_score`: Beta-distributed (skewed toward 0) so most rows
  look human with a believable bot-heavy tail, range [0, 1].
- `product_surface`: drawn from a product-specific list of plausible surfaces
  (e.g. Facebook: Feed/Groups/Marketplace/Video; Instagram: Feed/Reels/Stories/Explore;
  Threads: Feed/Profile/Search).

## Per-product schema drift

| | Facebook | Instagram | Threads |
|---|---|---|---|
| id column | `event_id` (UUID string) | `event_id` (int64, sequential) | `evt_id` (UUID string — different column name) |
| user column | `user_id` (int) | `uid` (string, e.g. `"ig_00234"`) | `account_id` (int) |
| timestamp | ISO 8601 string | **Unix epoch int** (seconds) | ISO 8601 string |
| country format | ISO-2 (`"US"`) | mixed ISO-2 / full name (`"US"`, `"United States"`) | ISO-3 (`"USA"`) |
| duration column | `session_duration_seconds` (float) | `session_duration_seconds` (int, truncated to whole seconds) | `session_duration_seconds` (float, null for non-session event types) |
| column order | canonical order | canonical order | shuffled relative to the other two |

All three still carry: id column, user column, `event_type`, timestamp column,
country column, duration column, `bot_probability_score`, `product_surface` —
just under product-specific names/types/order as above.

## Injected messiness

Applied independently per product file, at roughly 1–3% of rows each:

- Duplicate id values (a few rows re-emitted with the same id).
- Nulls in the country column and the duration column.
- A few out-of-order timestamps (a row stamped earlier than the rows around it).
- A few negative or implausibly large `session_duration_seconds` outliers.

## Structure

- One parametrized row-generator function, called once per product with a
  small per-product config (column names, id style, country format, surface
  list, timestamp encoding, column order).
- Each call returns a `pandas.DataFrame`; messiness is applied to that
  DataFrame before writing.
- `main()` generates all three DataFrames and writes them via
  `df.to_parquet(...)` (pyarrow engine) to:
  - `data/raw/facebook.parquet`
  - `data/raw/instagram.parquet`
  - `data/raw/threads.parquet`
- Prints row counts per file on completion.

## Dependencies

`pandas`, `numpy`, `pyarrow` — added to the currently-empty `requirements.txt`.

## Out of scope

- Loading this data into the warehouse (`etl/load_warehouse.py`) — separate,
  already-scaffolded piece of the pipeline.
- Any validation/gate logic — consumes this data but is not part of this change.
