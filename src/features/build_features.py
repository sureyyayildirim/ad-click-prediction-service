import numpy as np
import pandas as pd

def apply_feature_engineering(df):
    """
    Kişi 2 tarafından hazırlanan özellik mühendisliği adımı.
    Yaş verisini kategorik sepetlere (buckets) ayırır.
    """
    if "Age" in df.columns:
        # Age sütununu -np.inf ile np.inf arasında 3 farklı sepete böler
        df["Age_Bucket"] = pd.cut(
            df["Age"], bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0, 1, 2]
        ).astype(int)
        print("[Kişi 2] Problem Reframing: Age Bucketized.")
    return df
    
