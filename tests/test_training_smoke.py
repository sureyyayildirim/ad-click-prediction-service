import pandas as pd
import numpy as np
import pytest
import os
import mlflow
from src.features.prepare_dataset import build_features
from src.training.train_model import train_full_pipeline
import pytest

pytest.importorskip("xgboost")


@pytest.mark.smoke
def test_full_pipeline_flow(tmp_path, monkeypatch):
    """
    Can ve Yaren'in kodlarını (feature engineering + training)
    sahte veri üzerinden uçtan uca test eder.
    """
    # 1. MLflow'u geçici bir dizine yönlendir (Local ortamı kirletmemek için)
    tracking_dir = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tracking_dir}")

    # 2. prepare_dataset.py kodunun beklediği formatta sahte ham veri oluştur
    raw_data = pd.DataFrame(
        {
            "Daily Time Spent on Site": np.random.uniform(30, 90, 10),
            "Age": np.random.randint(18, 60, 10),
            "Area Income": np.random.uniform(20000, 80000, 10),
            "Daily Internet Usage": np.random.uniform(100, 300, 10),
            "Male": np.random.choice([0, 1], 10),
            "Ad Topic Line": [f"Ad Topic {i}" for i in range(10)],
            "City": [f"City {i}" for i in range(10)],
            "Country": [f"Country {i}" for i in range(10)],
            "Timestamp": pd.date_range(start="2023-01-01", periods=10, freq="H").astype(
                str
            ),
            "Clicked on Ad": np.random.choice([0, 1], 10),
        }
    )

    # 3. FEATURE ENGINEERING TESTİ
    # build_features fonksiyonunun çökmediğini ve beklenen sütunları ürettiğini doğrular
    try:
        processed_df = build_features(raw_data)
    except Exception as e:
        pytest.fail(f"build_features fonksiyonu hata verdi: {e}")

    assert "is_weekend" in processed_df.columns
    assert any("hash" in col for col in processed_df.columns)

    # 4. MODEL EĞİTİM TESTİ
    # Veriyi eğitim, validasyon ve test olarak ayır (Hızlı olması için küçük set)
    X = processed_df.drop("Clicked on Ad", axis=1)
    y = processed_df["Clicked on Ad"]

    # train_full_pipeline'ın beklediği 6 parçalı veri yapısı
    X_train, X_val, X_test = X[:6], X[6:8], X[8:]
    y_train, y_val, y_test = y[:6], y[6:8], y[8:]

    try:
        rf, xgb, ensemble = train_full_pipeline(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
    except Exception as e:
        pytest.fail(f"train_full_pipeline fonksiyonu hata verdi: {e}")

    # 5. ÇIKTI KONTROLÜ
    # Modellerin gerçekten oluştuğunu ve tahmin yapabildiğini doğrular
    assert ensemble is not None
    assert hasattr(ensemble, "predict")

    print("Smoke test başarıyla tamamlandı!")
