
import joblib

def save_model(model, scaler):
    joblib.dump(model, "models/churn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
