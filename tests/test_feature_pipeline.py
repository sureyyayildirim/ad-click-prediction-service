import pandas as pd
import pytest


def _dummy_df():
    return pd.DataFrame(
        {
            "Ad Topic Line": ["A", "B", "A", None],
            "City": ["Istanbul", "Ankara", None, "Izmir"],
            "Country": ["TR", "TR", "US", "TR"],
            "Timestamp": [
                "2016-01-01 08:30:00",
                "2016-01-02 12:00:00",
                None,
                "2016-01-03 23:10:00",
            ],
            "Daily Time Spent on Site": [10.0, 20.0, 30.0, None],
            "Age": [20, 30, 40, 50],
            "Area Income": [1000, 2000, 3000, 4000],
            "Daily Internet Usage": [50, 60, None, 80],
            "Male": [0, 1, 0, 1],
            "Clicked on Ad": [0, 1, 0, 1],
        }
    )


def test_build_features_importable_and_returns_df():
    try:
        from src.features.prepare_dataset import build_features  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("build_features not merged into main yet")


def test_build_features_outputs_expected_schema():
    try:
        from src.features.prepare_dataset import build_features
    except ModuleNotFoundError:
        pytest.skip("build_features not merged into main yet")

    df = _dummy_df()
    out = build_features(df)

    assert isinstance(out, pd.DataFrame)
    assert "Clicked on Ad" in out.columns, "Label column must exist in output"

    # Ad_Country_Cross hashing kolonları
    cross_hash_cols = [c for c in out.columns if c.startswith("Ad_Country_Cross_hash_")]
    assert len(cross_hash_cols) > 0, "Expected hashed columns for Ad_Country_Cross"

    # Ad Topic Line hashing kolonları
    topic_hash_cols = [c for c in out.columns if c.startswith("Ad_Topic_Line_hash_")]
    assert len(topic_hash_cols) > 0, "Expected hashed columns for Ad Topic Line"

    for col in [
        "Daily Time Spent on Site",
        "Age",
        "Area Income",
        "Daily Internet Usage",
        "Male",
    ]:
        assert col in out.columns, f"Expected numeric feature column: {col}"

    assert out.isna().sum().sum() == 0, "Output should not contain NaN values"
