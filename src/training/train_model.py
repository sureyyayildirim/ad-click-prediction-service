import mlflow
import mlflow.sklearn
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def train_full_pipeline(X_train_res, y_train_res, X_val, y_val, X_test, y_test):
    """
    Kişi 2 & 3: Model eğitimi, Checkpoint yönetimi, MLflow parametre ve artifact loglama.
    """
    # Parent run altında çalışabilmesi için nested=True
    with mlflow.start_run(run_name="MLOps_Level2_Pipeline", nested=True):
        # --- PARAMETRE LOGLAMA ---
        mlflow.log_params(
            {
                "upsampling": "True",
                "ensemble": "VotingSoft",
                "checkpoint_used": "True",
                "rf_max_depth": 10,
                "xgb_learning_rate": 0.1,
            }
        )

        # 1. MODEL A: Bagging (RF) - Kişi 2
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train_res, y_train_res)

        # 2. MODEL B: Boosting (XGB) with CHECKPOINT - Kişi 2 & 3
        checkpoint_path = "xgb_checkpoint.pkl"

        xgb_initial = XGBClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
        xgb_initial.fit(X_train_res, y_train_res)

        # Checkpoint Kaydı
        joblib.dump(xgb_initial, checkpoint_path)
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        # DÜZELTME: Modeli joblib ile geri yüklüyoruz
        loaded_xgb = joblib.load(checkpoint_path)

        xgb_final = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        # loaded_xgb.get_booster() diyerek XGBoost'un anladığı ham modeli veriyoruz
        xgb_final.fit(X_train_res, y_train_res, xgb_model=loaded_xgb.get_booster())

        mlflow.log_param("checkpoint_resumed", True)
        print(f"[Checkpoint] XGBoost model resumed from: {checkpoint_path}")

        # 3. MODEL C: ENSEMBLE (Voting) - Kişi 2
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb_final)], voting="soft"
        )
        ensemble.fit(X_train_res, y_train_res)

        # 4. VALIDATION CHECK
        val_acc = accuracy_score(y_val, ensemble.predict(X_val))
        print(f"\n[Validation] Ensemble Model Validation Accuracy: {val_acc:.4f}")
        mlflow.log_metric("val_accuracy", val_acc)

        # 5. METRIC LOGGING
        preds = ensemble.predict(X_test)
        probs = ensemble.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "auc_roc": roc_auc_score(y_test, probs),
        }

        print("\n" + "-" * 40)
        print("MLFLOW LOGGING: FINAL ENSEMBLE METRICS")
        print("-" * 40)
        for name, val in metrics.items():
            mlflow.log_metric(name, val)
            print(f"{name.upper():<10} : {val:.4f}")
        print("-" * 40)

        # 6. MODEL REGISTRY
        mlflow.sklearn.log_model(
            ensemble, "final_model", registered_model_name="AdClickPredictionModel"
        )

        return rf, xgb_final, ensemble
        
