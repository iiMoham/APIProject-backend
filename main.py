import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import PersonalityNote
import numpy as np
import pandas as pd
import pickle

# Create the app object
app = FastAPI(title="Personality Predictor API", version="1.0.0")

# ---------------------------------------------------------------------------
# CORS Configuration
# In production, replace "*" with your frontend server's URL, e.g.:
#   allow_origins=["https://your-frontend-domain.com"]
# ---------------------------------------------------------------------------
cors_origins_env = os.getenv("CORS_ORIGINS", "")
allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

# Fallback defaults for local testing and GitHub Pages frontend.
if not allowed_origins:
    allowed_origins = [
        "https://iimoham.github.io",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model (expects classifier.pkl in the same directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ensemble_model.pkl")

with open(MODEL_PATH, "rb") as f:
    classifier = pickle.load(f)


@app.get("/")
def index():
    return {"message": "Personality Predictor API is running. POST to /predict."}


@app.post("/predict")
def predict_personality(data: PersonalityNote):
    payload = data.dict()
    features = [[
        payload["Time_spent_Alone"],
        payload["Stage_fear"],
        payload["Social_event_attendance"],
        payload["Going_outside"],
        payload["Drained_after_socializing"],
        payload["Friends_circle_size"],
        payload["Post_frequency"],
    ]]

    prediction = classifier.predict(features)

    if prediction[0] > 0.5:
        result = "You must have a great connections bro!"
    else:
        result = "Get out of your coach you lazy guy"

    return {"prediction": result}


# Run with: python main.py
# Or with uvicorn directly: uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
