import numpy as np
import pytest


@pytest.mark.smoke
def test_train_full_pipeline_smoke(tmp_path, monkeypatch):
    try:
        from src.training.train_model import train_full_pipeline
    except ModuleNotFoundError:
        pytest.skip("train_full_pipeline is not merged into main yet")

    tracking_dir = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tracking_dir}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "ci-smoke")

    rng = np.random.default_rng(42)
    n_train, n_val, n_test, n_feat = 60, 20, 20, 8

    X_train_res = rng.normal(size=(n_train, n_feat))
    y_train_res = rng.integers(0, 2, size=n_train)

    X_val = rng.normal(size=(n_val, n_feat))
    y_val = rng.integers(0, 2, size=n_val)

    X_test = rng.normal(size=(n_test, n_feat))
    y_test = rng.integers(0, 2, size=n_test)

    rf, xgb_final, ensemble = train_full_pipeline(
        X_train_res, y_train_res, X_val, y_val, X_test, y_test
    )
    assert rf is not None and xgb_final is not None and ensemble is not None
