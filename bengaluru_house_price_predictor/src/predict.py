"""Load saved pipeline and predict one house.

The pipeline already contains imputer + OneHotEncoder, so pass RAW values
in the same column names as training.
"""
import joblib
import pandas as pd
from pathlib import Path

# absolute path -> works no matter where you run from
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "bengaluru_house_price_model.pkl")


def predict_one(area_type, availability, location, society,
                total_sqft, bath, balcony, bhk,
                model_path: str = MODEL_PATH) -> float:
    model = joblib.load(model_path)
    row = pd.DataFrame([{
        "area_type": area_type,
        "availability": availability,
        "location": location,
        "society": society if society else "Unknown",
        "total_sqft": total_sqft,
        "bath": bath,
        "balcony": balcony,
        "bhk": bhk,
        "sqft_per_bhk": total_sqft / bhk,
    }])
    return float(model.predict(row)[0])


if __name__ == "__main__":
    # Whitefield 2BHK example from your CSV
    p = predict_one("Super built-up  Area", "Ready To Move", "Whitefield",
                    "DuenaTa", 1170, 2, 1, 2)
    print(f"predicted: Rs {p:.2f} lakh")
