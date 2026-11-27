"""Feature generation for day-ahead price prediction.

`build_features` takes the scheduler inputs from Section 3 of the handout and
returns one row of features per hour. Training and inference both call it, so
the model can only ever be trained on information the scheduler will have.
"""
import numpy as np
import pandas as pd

T = 24
LOOKBACK = 7

# Supply-curve settings, tuned on walk-forward error (see tune.py).
BASE = "curve_near_hour"
HOUR_WIDTH = 0.75    # how far neighbouring hours count, in hours
LOAD_WIDTH = 2500.0  # how far other net-load levels count, in MW
RECENCY = 0.7        # weight multiplier per day into the past


def load_days(csv_path):
    """Read the training CSV into a dict of per-day arrays (n_days x 24)."""
    df = pd.read_csv(csv_path)
    if len(df) % T:
        raise ValueError("CSV does not contain whole 24-hour days")
    hours = df["hour"].to_numpy().reshape(-1, T)
    if not (hours == np.arange(1, T + 1)).all():
        raise ValueError("hours are not 1..24 in order for every day")
    if df.isna().any().any():
        raise ValueError("CSV contains missing values")

    days = {
        "month": df["month"].to_numpy()[::T],
        "day": df["day_of_month"].to_numpy()[::T],
        "weekday": df["day_of_week"].to_numpy()[::T],
        "forecast": df["da_net_load_forecast_mw"].to_numpy(float).reshape(-1, T),
        "net_load": df["realized_net_load_mw"].to_numpy(float).reshape(-1, T),
        "price": df["da_price_usd_per_mwh"].to_numpy(float).reshape(-1, T),
    }
    # The lag features assume consecutive days.
    wd = days["weekday"]
    if not (wd[1:] == wd[:-1] % 7 + 1).all():
        raise ValueError("days are not consecutive")
    return days


def scheduler_inputs(days, i):
    """Arguments the evaluator would pass for day i (24x7 pasts, yesterday first)."""
    if i < LOOKBACK:
        raise ValueError(f"day {i} has fewer than {LOOKBACK} preceding days")
    past = [i - k for k in range(1, LOOKBACK + 1)]
    return dict(
        month=int(days["month"][i]),
        day=int(days["day"][i]),
        weekday=int(days["weekday"][i]),
        net_load_forecast=days["forecast"][i],
        past_net_load_forecasts=days["forecast"][past].T,
        past_realized_net_load=days["net_load"][past].T,
        past_realized_da_prices=days["price"][past].T,
    )


def _supply_curve(load, price, weight, query, bandwidth):
    """Local-linear fit of price against net load, evaluated at each query load."""
    out = np.empty(len(query))
    for j, q in enumerate(query):
        d = load - q
        w = weight * np.exp(-0.5 * (d / bandwidth) ** 2) + 1e-6 * weight
        sw, sd = w.sum(), (w * d).sum()
        sdd = (w * d * d).sum() + 0.1 * sw * bandwidth**2  # damp extrapolation
        sp, sdp = (w * price).sum(), (w * d * price).sum()
        out[j] = (sdd * sp - sd * sdp) / (sw * sdd - sd * sd)
    return out


def build_features(month, day, weekday, net_load_forecast, past_net_load_forecasts,
                   past_realized_net_load, past_realized_da_prices):
    """Return (X, names) where X is 24 x n_features, one row per hour.

    month and day are accepted to match the scheduler interface but unused: one
    year of data cannot separate seasonality from that year's one-off events.
    """
    f = np.asarray(net_load_forecast, float).reshape(T)
    pf = np.asarray(past_net_load_forecasts, float).reshape(T, LOOKBACK)
    pl = np.asarray(past_realized_net_load, float).reshape(T, LOOKBACK)
    pp = np.asarray(past_realized_da_prices, float).reshape(T, LOOKBACK)
    h = np.arange(T)
    cols = {}

    # Price implied by the past week's price-vs-net-load curve at today's forecast.
    # Past *forecasts* are used because DA prices clear on day-ahead information.
    recency = RECENCY ** np.arange(LOOKBACK)
    base = np.empty(T)
    for j in h:
        dist = np.abs(h - j)
        hour_w = np.exp(-0.5 * (np.minimum(dist, T - dist) / HOUR_WIDTH) ** 2)
        w = np.outer(hour_w, recency).ravel()
        base[j] = _supply_curve(pf.ravel(), pp.ravel(), w, f[j:j + 1], LOAD_WIDTH)[0]
    cols[BASE] = base

    # The same curve without the hour restriction, and from yesterday alone. These
    # and the other prices are offsets from the base, so they survive level shifts.
    all_w = np.tile(recency, T)
    cols["curve_all_hours"] = _supply_curve(pf.ravel(), pp.ravel(), all_w, f, 1500.0) - base
    cols["curve_yesterday"] = _supply_curve(pf[:, 0], pp[:, 0], np.ones(T), f, 2500.0) - base
    cols["price_lag1"] = pp[:, 0] - base
    cols["price_med7"] = np.median(pp, axis=1) - base
    cols["price_day_lag1"] = pp[:, 0].mean() - base
    cols["price_day_mean7"] = pp.mean() - base
    cols["price_std7"] = np.full(T, pp.std())

    # Net-load forecast (GW): level, ramp, change from recent days.
    prev, nxt = np.maximum(h - 1, 0), np.minimum(h + 1, T - 1)
    cols["load"] = f / 1000
    cols["load_ramp"] = (f[nxt] - f[prev]) / 1000
    cols["load_vs_lag1"] = (f - pf[:, 0]) / 1000
    cols["load_vs_week"] = (f - pf.mean(1)) / 1000

    # Recent forecast error (realized minus forecast, GW).
    cols["err_lag1"] = (pl[:, 0] - pf[:, 0]) / 1000
    cols["err_week"] = (pl - pf).mean(1) / 1000

    # Calendar (weekday: 1 = Sunday ... 7 = Saturday).
    cols["weekend"] = np.full(T, float(weekday in (1, 7)))
    cols["monday"] = np.full(T, float(weekday == 2))
    for k in (1, 2):
        cols[f"hour_sin{k}"] = np.sin(2 * np.pi * k * h / T)
        cols[f"hour_cos{k}"] = np.cos(2 * np.pi * k * h / T)

    return np.column_stack(list(cols.values())), list(cols)


def build_dataset(days):
    """Stack features and targets for every day with a full week of history.

    Returns X (n x 24 x k), Y (n x 24), feature names, and the day indices used.
    """
    idx = np.arange(LOOKBACK, len(days["price"]))
    feats = [build_features(**scheduler_inputs(days, i)) for i in idx]
    return np.stack([x for x, _ in feats]), days["price"][idx], feats[0][1], idx
