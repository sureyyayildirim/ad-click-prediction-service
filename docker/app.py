from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Ad Click Prediction Service",
    description="Stateless REST API for ad click prediction",
    version="1.0"
)

MODEL_PATH = "final_deployment_model.pkl"
model = joblib.load(MODEL_PATH)


class AdClickInput(BaseModel):
    Daily_Time_Spent_on_Site: float
    Age: int
    Area_Income: float
    Daily_Internet_Usage: float
    Male: int
    hour: int
    day_of_week: int
    is_weekend: int


@app.post("/predict")
def predict_click(data: AdClickInput):

    # 🔑 TEK SATIRLIK DOĞRU INPUT
    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "click_prediction": int(prediction),
        "click_probability": round(float(probability), 4)
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
