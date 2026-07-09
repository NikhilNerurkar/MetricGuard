import pandas as pd
import pytest
import duckdb

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
    # Verify the column is a proper datetime type, not a string/object
    assert result["first_seen_date"].dtype.kind == 'M'  # 'M' is datetime64 kind
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
