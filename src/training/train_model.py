import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def train_full_pipeline(X_train_res, y_train_res, X_val, y_val, X_test, y_test):
    # Parent run altında çalışabilmesi için nested=True
    with mlflow.start_run(run_name="MLOps_Level2_Pipeline", nested=True):
        # 1. MODEL A: Bagging (RF)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train_res, y_train_res)

        # 2. MODEL B: Boosting (XGB) - MacBook (ARM64) uyumlu yapılandırma
        xgb_final = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_final.fit(X_train_res, y_train_res)

        # MacBook hatasını önlemek için tip tanımlıyoruz
        xgb_final._estimator_type = "classifier"

        # Checkpoint Kaydı (Resilience Gereksinimi)
        checkpoint_path = "xgb_checkpoint.json"
        xgb_final.save_model(checkpoint_path)
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        # 3. MODEL C: ENSEMBLE (Voting)
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb_final)], voting="soft"
        )
        ensemble.fit(X_train_res, y_train_res)

        # 4. METRIC LOGGING (CI/CD Uyumluluğu: Alt tire kullanımı)
        preds = ensemble.predict(X_test)
        probs = ensemble.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "AUC_ROC": roc_auc_score(y_test, probs),  # Tire yerine alt tire
        }
        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        # 5. MODEL REGISTRY (Governance Gereksinimi: 1.17)
        mlflow.sklearn.log_model(
            ensemble, "final_model", registered_model_name="AdClickPredictionModel"
        )
        print("\n--- BİLGİ: Model MLflow Registry'ye başarıyla kaydedildi! ---")

        return rf, xgb_final, ensemble
