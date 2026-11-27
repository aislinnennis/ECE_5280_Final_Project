"""3-feature day-ahead price predictor (ECE 4800/5280 battery project).

Predicts the 24 hourly day-ahead prices ($/MWh) of the trading day from the same
inputs the scheduler receives. Only numpy is needed. The trained weights are built in
below, so no model file is required.

Model: pooled ridge regression (penalty 1) on standardized features, one set of weights
shared by all 24 hours:

    price[h] = b + sum_j w_j * (feature_j[h] - mean_j) / std_j

Features (computed only from the scheduler inputs, so no look-ahead):
    net_load_forecast  today's day-ahead net-load forecast at hour h (MW)
    price_yesterday    yesterday's realized day-ahead price at hour h ($/MWh)
    ramp_past_2h       forecast[h] - forecast[h-2]; hours 1-2 reach back into
                       yesterday's forecast (MW)

Trained on all days of training_data.csv except the first 7 (no full past week),
with Jan 11 hours 15-23 of realized load treated as missing (that does not affect
these three features). Out of sample (quarterly cross-validation) with our LP/MILP
scheduler: lossless $118,265 (96.1% of perfect foresight), lossy $106,403.

Usage:
    from price_predictor_3feature import predict_prices
    prices = predict_prices(net_load_forecast, past_net_load_forecasts, past_realized_da_prices)

    # or with the scheduler's full argument list:
    prices = predict_prices_from_inputs(month, day, weekday, initial_energy, days_remaining,
                                        net_load_forecast, past_net_load_forecasts,
                                        past_realized_net_load, past_realized_da_prices)

    # retrain from the CSV (gives the same weights):  python price_predictor_3feature.py training_data.csv

Past arrays are 24 x 7 with column 0 = yesterday (column 6 = seven days ago).
"""
import numpy as np

FEATURES = ["net_load_forecast", "price_yesterday", "ramp_past_2h"]
RIDGE = 1.0

# Trained weights (standardized-feature space)
WEIGHTS = np.array([8.98908238498077, 22.08878929815885, 0.696259703694661])
FEATURE_MEAN = np.array([16013.782787836582, 32.39223963324053, 0.5372562674094707])
FEATURE_STD = np.array([8934.598503310564, 33.26417482742417, 5485.562730297539])
INTERCEPT = 32.36040806987001


def build_features(net_load_forecast, past_net_load_forecasts, past_realized_da_prices):
    """(24, 3) matrix: one row per hour, columns in the order of FEATURES."""
    fc = np.asarray(net_load_forecast, dtype=float).reshape(24)
    past_fc = np.asarray(past_net_load_forecasts, dtype=float).reshape(24, 7)
    past_price = np.asarray(past_realized_da_prices, dtype=float).reshape(24, 7)
    both_days = np.concatenate([past_fc[:, 0], fc])        # yesterday's 24 h, then today's 24 h
    ramp_2h = fc - both_days[22:46]                        # forecast[h] - forecast[h-2]
    return np.column_stack([fc, past_price[:, 0], ramp_2h])


def predict_prices(net_load_forecast, past_net_load_forecasts, past_realized_da_prices,
                   weights=WEIGHTS, mean=FEATURE_MEAN, std=FEATURE_STD, intercept=INTERCEPT):
    """Predicted prices for the 24 hours of the trading day, shape (24,)."""
    X = build_features(net_load_forecast, past_net_load_forecasts, past_realized_da_prices)
    return ((X - mean) / std) @ weights + intercept


def predict_prices_from_inputs(month, day, weekday, initial_energy, days_remaining,
                               net_load_forecast, past_net_load_forecasts,
                               past_realized_net_load, past_realized_da_prices):
    """Same as predict_prices, but takes the scheduler's 9 inputs in order."""
    return predict_prices(net_load_forecast, past_net_load_forecasts, past_realized_da_prices)


def fit(X, y, ridge=RIDGE):
    """Ridge fit. X: (n_days, 24, 3) features, y: (n_days, 24) prices.
    Returns (weights, mean, std, intercept). The intercept is not penalized."""
    X = X.reshape(-1, X.shape[-1])
    y = np.asarray(y, dtype=float).reshape(-1)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    w = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return w, mean, std, y.mean()


def training_set(csv_path):
    """Features and targets for every day with a full past week (day 8 onward).
    CSV columns: month, day_of_month, day_of_week, hour, da_net_load_forecast_mw,
    realized_net_load_mw, da_price_usd_per_mwh (24 rows per day, in order)."""
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    fc = raw[:, 4].reshape(-1, 24)
    price = raw[:, 6].reshape(-1, 24)
    X, y = [], []
    for d in range(7, len(fc)):
        past_fc = fc[d - 7:d][::-1].T          # (24, 7), column 0 = yesterday
        past_price = price[d - 7:d][::-1].T
        X.append(build_features(fc[d], past_fc, past_price))
        y.append(price[d])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("usage: python price_predictor_3feature.py training_data.csv")
    X, y = training_set(sys.argv[1])
    w, mean, std, b = fit(X, y)
    pred = ((X.reshape(-1, 3) - mean) / std) @ w + b
    print(f"trained on {len(y)} days")
    print("weights  ", w.tolist())
    print("mean     ", mean.tolist())
    print("std      ", std.tolist())
    print("intercept", float(b))
    print(f"in-sample RMSE ${np.sqrt(np.mean((pred - y.reshape(-1)) ** 2)):.2f}/MWh")
    print("max |difference| from built-in weights:", float(np.abs(w - WEIGHTS).max()))
