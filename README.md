# Backend — Personality Predictor API

A FastAPI application that serves an ML model to predict personality type (introvert/extrovert) from behavioral inputs.

## Prerequisites

- Python 3.9+
- `classifier.pkl` model file (copy from the project root)

## Setup

```bash
# 1. Copy the model file into this folder (one-time step)
copy ..\classifier.pkl .\classifier.pkl     # Windows
# cp ../classifier.pkl ./classifier.pkl     # Linux/macOS

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://<server-ip>:8000`.

## Endpoints

| Method | Path       | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | Health-check / welcome message     |
| POST   | `/predict` | Run personality prediction         |

### POST `/predict` — Request Body

```json
{
  "Time_spent_Alone": 5.0,
  "Stage_fear": 3.0,
  "Social_event_attendance": 2.0,
  "Going_outside": 4.0,
  "Drained_after_socializing": 8.0,
  "Friends_circle_size": 3.0,
  "Post_frequency": 1.0
}
```

### Response

```json
{ "prediction": "Get out of your coach you lazy dump ass" }
```

## Production Notes

- **CORS**: Change `allow_origins=["*"]` in `main.py` to your frontend's exact URL.
- **Process manager**: Use `gunicorn` + `uvicorn` workers or a systemd service in production.
- **Environment variables**: Consider moving model path and allowed origins to a `.env` file.
- **LightGBM runtime dependency**: If startup fails with `OSError: libgomp.so.1`, use the included `Dockerfile` (installs `libgomp1`) or install `libgomp1` in your host/container image.

## Docker Deploy (Recommended)

```bash
docker build -t personality-api .
docker run -p 8000:8000 personality-api
```
