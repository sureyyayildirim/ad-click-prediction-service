import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

print("Starting Data Engineering Pipeline...")

INPUT_FILE = "data/advertising.csv"
OUTPUT_FILE = "data/processed_adv_data.csv"

# Hashing Configuration
# 'Ad Topic Line' has ~1000 unique values, using 128 buckets.
HASH_FEATURES = {
    "Ad Topic Line": 128,
    "City": 32,
    "Country": 16,
    "Ad_Country_Cross": 64
}

NUMERIC_COLS = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male"
]

LABEL_COL = "Clicked on Ad"

# 1. LOAD DATA
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: '{INPUT_FILE}' not found!")
    print("Please make sure the file name is correct and it is in the same directory.")
    raise

# SAFETY NET: HANDLING MISSING VALUES
print("Checking and handling missing values...")
for col in NUMERIC_COLS:
    if col in df.columns:
        # Fill numeric missing values with the mean
        df[col] = df[col].fillna(df[col].mean())

# Fill missing text values with 'Unknown' to prevent hashing errors
text_cols = ["Ad Topic Line", "City", "Country"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# 2. FEATURE ENGINEERING (TIME)
# Parsing timestamp into usable features
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["hour"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Drop original timestamp column
    df.drop(columns=["Timestamp"], inplace=True)

# 3. FEATURE CROSS
# Combining Ad Topic Line and Country
df["Ad_Country_Cross"] = (
    df["Ad Topic Line"].astype(str) + "_" + df["Country"].astype(str)
)
print("Feature Cross: 'Ad_Country_Cross' created.")

# 4. HASHING FUNCTION (SKLEARN)
def hash_column(series, n_features, prefix):
    """
    Applies the Hashing Trick using sklearn's FeatureHasher.
    """
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    tokens = series.astype(str).apply(lambda x: [x])
    hashed = hasher.transform(tokens)

    # Convert to DataFrame
    hashed_df = pd.DataFrame(
        hashed.toarray(),
        columns=[f"{prefix}_hash_{i}" for i in range(n_features)]
    )
    return hashed_df

# Applying Hashing
hashed_dfs = []
print("Applying Hashing Trick (using sklearn FeatureHasher)...")

for col, n_features in HASH_FEATURES.items():
    if col in df.columns:
        hashed_df = hash_column(df[col], n_features, col.replace(" ", "_"))
        hashed_dfs.append(hashed_df)
    else:
        print(f"Column '{col}' not found, skipping.")

# 5. NUMERICAL SCALING
# Standardizing numerical features (Mean=0, Std=1)
num_df = df[NUMERIC_COLS].copy()
scaler = StandardScaler()

existing_num_cols = [c for c in NUMERIC_COLS if c in df.columns]
num_df_scaled = pd.DataFrame(
    scaler.fit_transform(num_df[existing_num_cols]),
    columns=existing_num_cols
)

# 6. FINAL DATASET CONSTRUCTION
label_df = df[[LABEL_COL]].copy()

time_cols = ["hour", "day_of_week", "is_weekend"]
existing_time_cols = [c for c in time_cols if c in df.columns]

# Concatenate all parts
final_df = pd.concat(
    [num_df_scaled] + hashed_dfs + [df[existing_time_cols], label_df],
    axis=1
)

# 7. METADATA & STATISTICS
# Generating a baseline statistics report
feature_stats = {}
for col in final_df.columns:
    if col != LABEL_COL:
        feature_stats[col] = {
            "mean": float(final_df[col].mean()),
            "std": float(final_df[col].std()),
            "min": float(final_df[col].min()),
            "max": float(final_df[col].max())
        }

stats_df = pd.DataFrame(feature_stats).T
stats_df.to_csv("data/feature_baseline_stats.csv")
print("Data statistics saved to 'data/feature_baseline_stats.csv'.")

# 8. SAVE OUTPUT
final_df.to_csv(OUTPUT_FILE, index=False)

print("-" * 40)
print(f"PROCESS COMPLETE! Output file ready: {OUTPUT_FILE}")
print(f"Final Shape: {final_df.shape}")
print("-" * 40)
