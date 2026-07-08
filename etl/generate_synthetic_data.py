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
