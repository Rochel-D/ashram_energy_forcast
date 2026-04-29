import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

# Load your combined dataset
df = pd.read_csv('data/combined_dataset.csv', parse_dates=['timestamp'])

# Preprocessing
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

for col in ['air_temperature', 'dew_temperature', 'wind_speed']:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

if 'primary_use' in df.columns:
    df = pd.get_dummies(df, columns=['primary_use'])

# Remove zero/ negative readings
df.drop(columns=['timestamp', 'building_id', 'site_id'], inplace=True, errors='ignore')
df = df[df['meter_reading'] > 0]

# Drop all remaining NaN values 
df = df.dropna()
print("Shape after dropping NaN:", df.shape)

# Split features and target
X = df.drop(columns=['meter_reading'])
y = df['meter_reading']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLFlow setup
mlflow.set_tracking_uri("file:///Users/rocheldasilva/Desktop/ashrae_energy_forcast/mlruns")
mlflow.set_experiment("ashrae-energy-forecasting")

models = {
    "Linear_Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Random_Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient_Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "model")

        results[name] = {"RMSE": round(rmse, 4), "R2": round(r2, 4)}
        print(f"  ✅ RMSE: {rmse:.4f} | R2: {r2:.4f}")

# Save best model
results_df = pd.DataFrame(results).T.sort_values('RMSE')
print("\n📊 Model Comparison:")
print(results_df)

best_name = results_df.index[0]
print(f"\n🏆 Best Model: {best_name}")

os.makedirs('models', exist_ok=True)
joblib.dump(models[best_name], 'models/best_model.pkl')
print("✅ Best model saved to models/best_model.pkl")