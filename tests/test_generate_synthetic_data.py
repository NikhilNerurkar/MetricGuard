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
