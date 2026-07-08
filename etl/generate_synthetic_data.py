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
