import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# PROJE YOL AYARI
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Modül Importları
from src.features.build_features import apply_feature_engineering
from src.features.rebalancing import analyze_and_rebalance
from src.evaluation.before_after_analysis import run_before_after_comparison
from src.training.train_model import train_full_pipeline

# Önceki açık kalan run varsa kapat
if mlflow.active_run():
    mlflow.end_run()

# MLflow Deneyini Ayarla
mlflow.set_experiment("MLOps_Term_Project_Ad_Click")

# ANA PARENT RUN BAŞLANGICI
with mlflow.start_run(run_name="Full_MLOps_Pipeline_Flow"):

    # 1. VERİ HAZIRLIĞI
    data_file = os.path.join(BASE_DIR, 'processed_adv_data.csv')
    df = pd.read_csv(data_file)
    df = apply_feature_engineering(df) # Age Bucketizing

    # 2. SPLIT (80/10/10)
    X = df.drop('Clicked on Ad', axis=1)
    y = df['Clicked on Ad']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # 3. REBALANCING
    train_df = pd.concat([X_train, y_train], axis=1)
    upsampled_train = analyze_and_rebalance(train_df)
    X_train_res = upsampled_train.drop('Clicked on Ad', axis=1)
    y_train_res = upsampled_train['Clicked on Ad']

    # 4. BEFORE vs AFTER
    run_before_after_comparison(X_train, y_train, X_train_res, y_train_res, X_test, y_test)

    # Nested Run
    rf, xgb, ensemble = train_full_pipeline(X_train_res, y_train_res, X_val, y_val, X_test, y_test)

    # 6. PERFORMANCE TABLE
    results = []
    models_dict = {"Bagging (RF)": rf, "Boosting (XGB)": xgb, "Ensemble (Voting)": ensemble}

    for name, model in models_dict.items():
        p = model.predict(X_test)
        pr = model.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, p), 2),
            "Precision": round(precision_score(y_test, p), 6),
            "Recall": round(recall_score(y_test, p), 2),
            "F1 Score": round(f1_score(y_test, p), 6),
            "AUC-ROC": round(roc_auc_score(y_test, pr), 4)
        })

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("MLOps Project - Model Performance Table")
    print("=" * 90)
    print(results_df.to_string(index=False, justify='right'))
    print("=" * 90)

    # Visualization
    metrics_list = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'plum']
    for i, metric in enumerate(metrics_list):
        plt.figure(figsize=(8, 5))
        plt.bar(results_df['Model'], results_df[metric], color=colors[i], width=0.6)
        plt.title(f'Metric Comparison: {metric}')
        plt.ylim(0.7, 1.05)
        plt.show()

    # 8. MODEL KAYDI
    save_path = os.path.join(BASE_DIR, "final_deployment_model.pkl")
    joblib.dump(ensemble, save_path)
    print(f"\nSüreç tamamlandı. Model: {save_path}")