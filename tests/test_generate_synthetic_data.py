from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from etl.generate_synthetic_data import (
    _generate_canonical_events,
    EVENT_TYPES,
    EVENT_TYPE_WEIGHTS,
    LOOKBACK_DAYS,
    PRODUCT_CONFIGS,
)

FIXED_NOW = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


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
