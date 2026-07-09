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
