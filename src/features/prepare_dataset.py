import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
import joblib
import os

INPUT_FILE = "data/advertising.csv"
OUTPUT_FILE = "data/processed_adv_data.csv"
STATS_FILE = "data/feature_baseline_stats.csv"
SCALER_FILE = "data/scaler.joblib"

HASH_FEATURES = {
    "Ad Topic Line": 128,
    "City": 32,
    "Country": 16,
    "Ad_Country_Cross": 64,
}

NUMERIC_COLS = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
]
LABEL_COL = "Clicked on Ad"


def hash_column(series, n_features, prefix):

    hasher = FeatureHasher(n_features=n_features, input_type="string")
    tokens = series.astype(str).apply(lambda x: [x])
    hashed = hasher.transform(tokens)
    return pd.DataFrame(
        hashed.toarray(), columns=[f"{prefix}_hash_{i}" for i in range(n_features)]
    )


def build_features(df: pd.DataFrame, is_training=True) -> pd.DataFrame:
    df = df.copy()
    df.reset_index(drop=True, inplace=True)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["hour"] = df["Timestamp"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["Timestamp"].dt.dayofweek.fillna(0).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df.drop(columns=["Timestamp"], inplace=True)

    if "Ad Topic Line" in df.columns and "Country" in df.columns:
        df["Ad_Country_Cross"] = (
            df["Ad Topic Line"].astype(str) + "_" + df["Country"].astype(str)
        )

    # 4. Hashing
    hashed_dfs = []
    for col, n_features in HASH_FEATURES.items():
        if col in df.columns:
            hashed_df = hash_column(df[col], n_features, col.replace(" ", "_"))
            hashed_dfs.append(hashed_df)

    # 5. Scaling
    existing_num_cols = [c for c in NUMERIC_COLS if c in df.columns]

    if is_training:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[existing_num_cols])
        joblib.dump(scaler, SCALER_FILE)
        print(f"Scaler saved to {SCALER_FILE}")
    else:
        if os.path.exists(SCALER_FILE):
            scaler = joblib.load(SCALER_FILE)
            scaled_values = scaler.transform(df[existing_num_cols])
        else:
            raise FileNotFoundError("Error.")

    num_df_scaled = pd.DataFrame(scaled_values, columns=existing_num_cols)

    time_cols = ["hour", "day_of_week", "is_weekend"]
    existing_time_cols = [c for c in time_cols if c in df.columns]

    dfs_to_concat = [num_df_scaled] + hashed_dfs + [df[existing_time_cols]]

    if is_training and LABEL_COL in df.columns:
        dfs_to_concat.append(df[[LABEL_COL]])

    return pd.concat(dfs_to_concat, axis=1)


if __name__ == "__main__":
    raw_df = pd.read_csv(INPUT_FILE)
    processed_df = build_features(raw_df, is_training=True)
