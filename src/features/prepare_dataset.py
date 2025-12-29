import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "data/advertising.csv"
OUTPUT_FILE = "data/processed_adv_data.csv"
STATS_FILE = "data/feature_baseline_stats.csv"

# Hashing Configuration
# 'Ad Topic Line' has ~1000 unique values, using 128 buckets.
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


# HASHING FUNCTION
def hash_column(series, n_features, prefix):
    """
    Applies the Hashing Trick using sklearn's FeatureHasher.
    """
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    tokens = series.astype(str).apply(lambda x: [x])
    hashed = hasher.transform(tokens)

    # Convert to DataFrame
    hashed_df = pd.DataFrame(
        hashed.toarray(), columns=[f"{prefix}_hash_{i}" for i in range(n_features)]
    )
    return hashed_df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Missing Values
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    text_cols = ["Ad Topic Line", "City", "Country"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # 2. Time Features
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["hour"] = df["Timestamp"].dt.hour
        df["day_of_week"] = df["Timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df.drop(columns=["Timestamp"], inplace=True)

    # 3. Feature Cross
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
    scaler = StandardScaler()
    num_df_scaled = pd.DataFrame(
        scaler.fit_transform(df[existing_num_cols]), columns=existing_num_cols
    )

    # 6. Merge All
    time_cols = ["hour", "day_of_week", "is_weekend"]
    existing_time_cols = [c for c in time_cols if c in df.columns]

    dfs_to_concat = [num_df_scaled] + hashed_dfs + [df[existing_time_cols]]
    if LABEL_COL in df.columns:
        dfs_to_concat.append(df[[LABEL_COL]])

    final_df = pd.concat(dfs_to_concat, axis=1)

    return final_df


def save_statistics(df, output_path):
    feature_stats = {}
    for col in df.columns:
        if col != LABEL_COL:
            feature_stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
    stats_df = pd.DataFrame(feature_stats).T
    stats_df.to_csv(output_path)
    print(f"Data statistics saved to '{output_path}'.")


# --- MAIN BLOCK ---
if __name__ == "__main__":
    print("Starting Data Engineering Pipeline...")

    try:
        # 1. Load
        raw_df = pd.read_csv(INPUT_FILE)
        print(f"Data loaded successfully. Shape: {raw_df.shape}")

        # 2. Transform (Using the function)
        processed_df = build_features(raw_df)

        # 3. Save Stats
        save_statistics(processed_df, STATS_FILE)

        # 4. Save Output
        processed_df.to_csv(OUTPUT_FILE, index=False)

        print("-" * 40)
        print(f"PROCESS COMPLETE! Output file ready: {OUTPUT_FILE}")
        print(f"Final Shape: {processed_df.shape}")
        print("-" * 40)

    except FileNotFoundError:
        print(f"ERROR: '{INPUT_FILE}' not found! Please check data/ directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
