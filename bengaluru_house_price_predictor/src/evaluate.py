"""Metrics in one place.

MAE  = average |error| in lakhs -> 'off by Rs X lakh on average'
RMSE = penalises big misses more
R2   = fraction of price variance explained (1.0 = perfect, 0 = mean baseline)
"""
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
