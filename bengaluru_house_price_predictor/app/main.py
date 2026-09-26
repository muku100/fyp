"""FastAPI: UI + API in one server.

- GET /          -> UI (app/static/index.html)
- GET /health    -> {"ok": True}  (use this for health checks)
- GET /locations -> {"locations": [...top 300 by frequency...]} for the datalist
- POST /predict  -> {predicted_price_lakh}

Run (from the project root, NOT from inside app/):
    cd bengaluru_house_price_predictor && .venv/bin/python -m uvicorn app.main:app --reload --port 8000
    # UI:   http://127.0.0.1:8000/
    # docs: http://127.0.0.1:8000/docs
"""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "bengaluru_house_price_model.pkl"
DATA_PATH = ROOT / "data" / "raw" / "bengaluru_house_prices.csv"
INDEX = Path(__file__).resolve().parent / "static" / "index.html"

app = FastAPI(title="Bengaluru House Price AI")


class House(BaseModel):
    area_type: str = "Super built-up  Area"
    availability: str = "Ready To Move"
    location: str
    society: str | None = None
    total_sqft: float = Field(gt=100, lt=20000)
    bath: float = Field(ge=0, le=15)
    balcony: float = Field(ge=0, le=10)
    bhk: int = Field(ge=1, le=15)


@lru_cache(maxsize=1)
def _model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _locations() -> list[str]:
    try:
        df = pd.read_csv(DATA_PATH, usecols=["location"])
        return df["location"].dropna().str.strip().value_counts().head(300).index.tolist()
    except Exception:
        return ["Whitefield", "Marathahalli", "Electronic City", "HSR Layout"]


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(INDEX)


@app.get("/health")
def health():
    return {"ok": True, "try": "POST /predict or open GET /"}


@app.get("/locations")
def locations(q: str | None = None, limit: int = 50):
    locs = _locations()
    if q:
        q = q.strip().lower()
        locs = [loc for loc in locs if q in loc.lower()]
    return {"locations": locs[:limit]}


@app.post("/predict")
def predict(h: House):
    model = _model()
    society = (h.society or "Unknown").strip() or "Unknown"
    row = pd.DataFrame([{
        "area_type": h.area_type.strip(),
        "availability": h.availability.strip(),
        "location": h.location.strip(),
        "society": society,
        "total_sqft": h.total_sqft,
        "bath": h.bath,
        "balcony": h.balcony,
        "bhk": h.bhk,
        "sqft_per_bhk": h.total_sqft / h.bhk,
    }])
    price = float(model.predict(row)[0])
    return {"predicted_price_lakh": round(price, 2)}
