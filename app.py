from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Energy Load Forecasting API")

model = joblib.load("models/best_model.pkl")

class InputData(BaseModel):
    hour: int
    dayofweek: int
    month: int
    is_weekend: int
    air_temperature: float
    dew_temperature: float
    wind_speed: float
    square_feet: float
    year_built: float
    floor_count: float

@app.get("/")
def home():
    return {"message": "⚡ Energy Forecasting API is running!"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {
        "predicted_energy_load_kwh": round(float(prediction), 4)
    }