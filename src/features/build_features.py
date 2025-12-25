import numpy as np
import pandas as pd

def apply_feature_engineering(df):
    if 'Age' in df.columns:
        df['Age_Bucket'] = pd.cut(df['Age'], bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0, 1, 2]).astype(int)
        print("[Kişi 2] Problem Reframing: Age Bucketized.")
    return df