# Bengaluru House Price ML

End-to-end regression: `location, bhk, sqft, bath, balcony, area_type, availability, society` → `price (lakhs)`.

Stack: **Pandas → scikit-learn (Pipeline/ColumnTransformer) + XGBoost (XGBRegressor) → FastAPI**

## What's doing what

```
data/raw/bengaluru_house_prices.csv  # SOURCE OF TRUTH, never edit
data/processed/cleaned.csv           # written by notebook 03, for inspection only

src/data_cleaning.py  # convert_sqft('2100 - 2850'->2475.0, '34.46Sq. Meter'->NaN),
                      # extract_bhk('2 BHK'->2, '4 Bedroom'->4, '1 RK'->1),
                      # strip ws, drop 529 dups, drop rows missing price/sqft/bhk
src/features.py       # add_features(): sqft_per_bhk = total_sqft/bhk (legit).
                      # FEATURES list (society INCLUDED). price_per_sqft is
                      # LEAKAGE -> EDA only, never in X.
src/preprocessing.py  # make_preprocessor(): numeric median-impute,
                      # categorical most_frequent-impute + OneHot(min_frequency=10,
                      # handle_unknown='ignore'). society missing->'Unknown',
                      # rare society/location grouped. No scaling (trees don't need it).
                      # make_ridge_preprocessor(): same + StandardScaler (linear needs it).
src/train.py          # load_data() -> split 80/20 (random_state=42) ->
                      # dummy -> ridge -> random_forest -> xgboost, prints MAE/RMSE/R2,
                      # saves best (XGBoost PIPELINE incl. imputer+encoder) to models/
src/evaluate.py       # regression_metrics(): MAE (avg |error| in lakhs),
                      # RMSE (punishes big misses), R2 (variance explained)
src/predict.py        # predict_one(...): loads models/*.pkl, builds 1-row DF incl.
                      # sqft_per_bhk, returns float lakhs. Pipeline handles raw strings.
app/main.py           # FastAPI: GET / (UI), GET /health, GET /locations,
                      # POST /predict -> {predicted_price_lakh} (model cached)
app/static/index.html # UI served at GET / (vanilla JS fetch, no build step)
models/bengaluru_house_price_model.pkl  # trained pipeline (1.5 MB)

notebooks/01_data_understanding.ipynb  # shape, missing, society/location cardinality
notebooks/02_eda.ipynb                 # price hist, outliers (price>1000: 45, bath>10: 20)
notebooks/03_feature_engineering.ipynb # cleaning demo, writes cleaned.csv
notebooks/04_baseline_models.ipynb     # dummy + ridge
notebooks/05_xgboost.ipynb             # RF vs XGB + feature importance
notebooks/06_tuning.ipynb              # RandomizedSearchCV grid (commented, slow)
notebooks/07_error_analysis.ipynb      # worst predictions, residuals
```

## How to run the project

Setup (once):
```
cd bengaluru_house_price_predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Learn (in order 01 → 07, run from the project root):
```
cd bengaluru_house_price_predictor && .venv/bin/python -m jupyter lab
# open notebooks/01_data_understanding.ipynb, run all cells, then 02, ...
```

Train + quick check (run from the project root):
```
cd bengaluru_house_price_predictor && .venv/bin/python -m src.train    # prints MAE/RMSE/R2, saves models/*.pkl
cd bengaluru_house_price_predictor && .venv/bin/python -m src.predict  # Whitefield 2BHK 1170sqft -> ~56.54 lakh
```

Serve API + UI (run from the project root — NOT from inside app/):
```
cd bengaluru_house_price_predictor && .venv/bin/python -m uvicorn app.main:app --reload --port 8000
# UI:   http://127.0.0.1:8000/        (form -> POST /predict via fetch)
# docs: http://127.0.0.1:8000/docs
# If you see "ModuleNotFoundError: No module named 'app'" you started it
# from the wrong folder (e.g. ~/projects/fyp). cd into
# bengaluru_house_price_predictor first — it is the folder that CONTAINS app/.
# Do NOT `cd app` (that breaks `app.main:app`).
```

## How to use the UI

1. Start the server (above), open http://127.0.0.1:8000/
2. Location has autocomplete (top 300 from `GET /locations`).
3. Fill sqft/bhk/bath/balcony, leave society empty if unknown, click **Predict price**.
4. `Fill Whitefield example` resets to the known-good test house (→ ~₹56.54 lakh).

## How to test the API

Health (`/` now serves the UI, so health moved to `/health`):
```
curl http://127.0.0.1:8000/health
# {"ok":true,"try":"POST /predict or open GET /"}
curl http://127.0.0.1:8000/ | head -c 100
# <!DOCTYPE html>... (the UI)
```

Predict (copy-paste):
```
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"location":"Whitefield","total_sqft":1170,"bath":2,"balcony":1,"bhk":2,"society":"DuenaTa","area_type":"Super built-up  Area","availability":"Ready To Move"}'
# {"predicted_price_lakh":56.54}
```

No-society case (missing -> "Unknown"):
```
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"location":"Marathahalli","total_sqft":1310,"bath":3,"balcony":1,"bhk":3}'
```

From Python (`requests`):
```python
import requests
r = requests.post("http://127.0.0.1:8000/predict", json={
    "location": "Whitefield", "total_sqft": 1170,
    "bath": 2, "balcony": 1, "bhk": 2,
    "society": "DuenaTa",
    "area_type": "Super built-up  Area",
    "availability": "Ready To Move",
})
print(r.status_code, r.json())  # 200 {'predicted_price_lakh': 56.54}
```

From Python (stdlib only, no `requests` needed):
```python
import json, urllib.request
body = json.dumps({
    "location": "Whitefield", "total_sqft": 1170,
    "bath": 2, "balcony": 1, "bhk": 2, "society": "DuenaTa",
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=body, headers={"Content-Type": "application/json"}, method="POST")
print(json.load(urllib.request.urlopen(req)))
```

From Postman:
```
1. Start the API: cd bengaluru_house_price_predictor && .venv/bin/python -m uvicorn app.main:app --reload --port 8000
2. UI check: new tab, GET http://127.0.0.1:8000/ -> Preview tab shows the form (HTML).
2. New Tab -> method POST -> URL http://127.0.0.1:8000/predict
3. Tab Body -> raw -> type JSON (dropdown on the right)
4. Paste:
   {"location":"Whitefield","total_sqft":1170,"bath":2,"balcony":1,"bhk":2,
    "society":"DuenaTa","area_type":"Super built-up  Area","availability":"Ready To Move"}
5. Send -> expect 200 OK + {"predicted_price_lakh":56.54}
Health check: new tab, GET http://127.0.0.1:8000/health -> {"ok":true,...}
If 422: click Body, check which field failed (total_sqft 100-20000, bath 0-15,
balcony 0-10, bhk 1-15, location is required).
Faster alternative with zero setup: open http://127.0.0.1:8000/docs in a
browser -> POST /predict -> Try it out -> Execute.
```

Rules: `total_sqft` 100–20000, `bath` 0–15, `balcony` 0–10, `bhk` 1–15.
Unknown location/society will NOT crash (`handle_unknown="ignore"`) — it just
falls back to the learned averages.

## How to retrain if data changes

1. Replace the CSV (keep the SAME 9 column names, run from the project root):
```
cd bengaluru_house_price_predictor && cp /path/to/new.csv data/raw/bengaluru_house_prices.csv
```
2. Sanity-check + retrain:
```
cd bengaluru_house_price_predictor && .venv/bin/python -c "import pandas as pd; df=pd.read_csv('data/raw/bengaluru_house_prices.csv'); print(df.shape, df.columns.tolist(), df.isnull().sum().to_dict())"
cd bengaluru_house_price_predictor && .venv/bin/python -m src.train
```
`src/train.py:load_data()` re-runs: fill society NaN→"Unknown" → clean →
engineer → 80/20 split → train 4 models → overwrite
`models/bengaluru_house_price_model.pkl`.
3. Verify the new model:
```
cd bengaluru_house_price_predictor && .venv/bin/python -m src.predict
# check MAE table printed by train: XGB should beat dummy (~78) and ridge (~45)
```
4. Restart the API (it loads the .pkl on every request, so just restart):
```
# Ctrl+C old uvicorn, then:
cd bengaluru_house_price_predictor && .venv/bin/python -m uvicorn app.main:app --reload --port 8000
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"location":"Whitefield","total_sqft":1170,"bath":2,"balcony":1,"bhk":2}'
```
5. If columns changed (e.g. renamed `bath`→`bathrooms`): update
`src/features.py:FEATURES/NUMERIC_FEATURES/CATEGORICAL_FEATURES`,
`src/data_cleaning.py`, and `app/main.py:House`, then retrain.

Tip: keep old models: `cp models/bengaluru_house_price_model.pkl models/v1_$(date +%F).pkl`
before retraining, so you can compare MAE old vs new.

## V1 results (13,320 rows → 12,728 after cleaning)
| model | MAE | RMSE | R2 |
|---|---|---|---|
| dummy(mean) | 78.21 | 168.17 | -0.00 |
| ridge | 44.80 | 119.77 | 0.49 |
| random_forest | 34.75 | 109.46 | 0.58 |
| xgboost | 34.06 | 110.17 | 0.57 |

MAE is in lakhs: XGB is off by ~Rs 34 lakh on average. Worst errors are
Rs 1000+ lakh outliers (e.g. Rs 3600 lakh house) — next step: outlier treatment.

## Key lessons baked into code
1. `society`: 41.3% missing, 2688 unique → fill `Unknown` + `OneHot(min_frequency=10)`.
2. `total_sqft='2100 - 2850'` → 2475.0; `'34.46Sq. Meter'`/Acres/Perch (46 rows) → NaN → dropped.
3. `size='2 BHK'/'4 Bedroom'/'1 RK'` → `bhk` number.
4. `price_per_sqft` is TARGET LEAKAGE — EDA only, never in X.
5. Trees (RF/XGB) need no scaling; Ridge does.
