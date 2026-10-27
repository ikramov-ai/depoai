
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="DepoAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ishlab chiqish uchun
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "depo_model_small.pkl"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day",
    "month", "campaign", "pdays", "previous", "poutcome"
]

class Customer(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.get("/")
def home():
    return {"message": "DepoAI API ishlayapti"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: Customer):
    try:
        data = pd.DataFrame([customer.model_dump()])
        data = data[FEATURES]

        prediction = int(model.predict(data)[0])

        result = {
            "prediction": prediction,
            "label": (
                "Depozitga rozilik bildirishi mumkin"
                if prediction == 1
                else "Depozitga rozilik bildirmasligi mumkin"
            )
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(data)[0]
            result["probability_yes"] = round(
                float(probabilities[list(model.classes_).index(1)]) * 100,
                2
            )

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
