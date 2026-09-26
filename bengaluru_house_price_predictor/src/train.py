"""Train + compare: Dummy -> Ridge -> RandomForest -> XGBoost.

Why in this order? Each one must BEAT the previous to justify its complexity.
XGBoost uses the sklearn-compatible XGBRegressor inside a Pipeline, so
raw fields (with missing society etc.) go in, price comes out.

Usage:
    cd bengaluru_house_price_predictor
    .venv/bin/python -m src.train
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data_cleaning import clean_dataframe
from src.evaluate import regression_metrics
from src.features import TARGET, FEATURES, add_features
from src.preprocessing import make_preprocessor, make_ridge_preprocessor

# absolute paths -> runnable from any cwd (as long as `src` is importable)
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(ROOT / "data" / "raw" / "bengaluru_house_prices.csv")
MODEL_PATH = str(ROOT / "models" / "bengaluru_house_price_model.pkl")


def load_data(path: str = DATA_PATH):
    df = pd.read_csv(path)
    # society missing -> explicit 'Unknown' BEFORE most_frequent impute,
    # so the model learns 'unknown society' as its own signal
    df["society"] = df["society"].fillna("Unknown")
    df = clean_dataframe(df)
    df = add_features(df)
    X = df[FEATURES]
    y = df[TARGET]
    return X, y, df


def build_models():
    ridge = Pipeline([
        ("preprocessor", make_ridge_preprocessor()),
        ("model", Ridge()),
    ])
    rf = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    xgb = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    dummy = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", DummyRegressor(strategy="mean")),
    ])
    return {"dummy": dummy, "ridge": ridge, "random_forest": rf, "xgboost": xgb}


def main():
    X, y, df = load_data()
    print(f"rows after cleaning: {len(df)} (raw 13320, dups + bad sqft/bhk removed)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    results = {}
    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        m = regression_metrics(y_test, pred)
        results[name] = m
        print(f"{name:13s} MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R2={m['r2']:.3f}")

    best = build_models()["xgboost"]
    best.fit(X_train, y_train)
    joblib.dump(best, MODEL_PATH)
    print(f"\nsaved XGBoost pipeline -> {MODEL_PATH}")
    print("pipeline includes imputer+encoder, so app can send RAW fields.")

    # worst predictions for error analysis
    out = pd.DataFrame({"actual": y_test, "predicted": best.predict(X_test)})
    out["abs_error"] = (out["predicted"] - out["actual"]).abs()
    print("\nTop 5 worst predictions:")
    print(out.sort_values("abs_error", ascending=False).head(5).to_string())


if __name__ == "__main__":
    main()
