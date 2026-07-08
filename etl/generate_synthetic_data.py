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
        df[_CANONICAL_DURATION_COL] = df[_CANONICAL_DURATION_COL].astype("int64")
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
        shifted_epochs = pd.Series(
            df.loc[reorder_idx, timestamp_col].values - shift_days * 86400,
            index=reorder_idx,
            dtype="int64"
        )
        df.loc[reorder_idx, timestamp_col] = shifted_epochs
    else:
        parsed = pd.to_datetime(df.loc[reorder_idx, timestamp_col])
        shifted = parsed - pd.to_timedelta(shift_days, unit="D")
        df.loc[reorder_idx, timestamp_col] = shifted.dt.strftime("%Y-%m-%dT%H:%M:%S")

    return df
