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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://iimoham.github.io/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model (expects classifier.pkl in the same directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "classifier.pkl")

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
        result = "Get out of your coach you lazy dump ass"

    return {"prediction": result}


# Run with: python main.py
# Or with uvicorn directly: uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
