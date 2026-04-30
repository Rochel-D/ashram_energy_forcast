from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Energy Load Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("/home/ubuntu/ashram_energy_forcast/models/best_model.pkl")
feature_names = model.feature_names_in_

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
    return {"message": "Energy Forecasting API is running!"}

@app.post("/predict")
def predict(data: InputData):
    # Create dataframe with all required features
    df = pd.DataFrame(columns=feature_names)
    df.loc[0] = 0  # Fill with zeros
    
    # Fill known features
    df['hour'] = data.hour
    df['dayofweek'] = data.dayofweek
    df['month'] = data.month
    df['is_weekend'] = data.is_weekend
    df['air_temperature'] = data.air_temperature
    df['dew_temperature'] = data.dew_temperature
    df['wind_speed'] = data.wind_speed
    df['square_feet'] = data.square_feet
    df['year_built'] = data.year_built
    df['floor_count'] = data.floor_count
    
    prediction = model.predict(df)[0]
    return {"predicted_energy_load_kwh": round(float(prediction), 4)}