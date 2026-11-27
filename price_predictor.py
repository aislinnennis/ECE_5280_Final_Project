"""Day-ahead price predictor: local supply curve + robust ridge correction.

    from price_predictor import predict_prices
    prices = predict_prices(month, day, weekday, net_load_forecast,
                            past_net_load_forecasts, past_realized_net_load,
                            past_realized_da_prices)      # 24 values, $/MWh

Retrain with:  python price_predictor.py path/to/training_data.csv
"""
import sys
from pathlib import Path

import numpy as np

from features import BASE, T, build_dataset, build_features, load_days

MODEL_FILE = Path(__file__).resolve().parent / "price_model.npz"


class PricePredictor:
    """Predicts price = supply-curve feature + linear correction.

    The correction minimises Huber loss with an L2 penalty, by iteratively
    reweighted least squares. Huber loss keeps a few $300-600 spike hours from
    dominating the fit; the penalty shrinks the correction toward zero.
    """

    def __init__(self, ridge=1000.0, huber_delta=5.0, n_iter=15):
        self.ridge = ridge
        self.huber_delta = huber_delta
        self.n_iter = n_iter

    def fit(self, X, Y, names):
        """X: n_days x 24 x k features, Y: n_days x 24 prices."""
        self.names = list(names)
        A = X.reshape(-1, X.shape[-1])
        b = self.names.index(BASE)
        y = Y.ravel() - A[:, b]

        self.mean, self.std = A.mean(0), A.std(0) + 1e-9
        Z = (A - self.mean) / self.std
        w = np.ones(len(y))
        for _ in range(self.n_iter):
            zbar, ybar = np.average(Z, axis=0, weights=w), np.average(y, weights=w)
            Zc = Z - zbar
            gram = Zc.T @ (Zc * w[:, None]) + self.ridge * np.eye(Z.shape[1])
            self.coef = np.linalg.solve(gram, Zc.T @ (w * (y - ybar)))
            self.intercept = ybar - zbar @ self.coef
            resid = np.abs(y - self.intercept - Z @ self.coef)
            w = np.minimum(1.0, self.huber_delta / np.maximum(resid, 1e-9))
        return self

    def predict(self, X):
        """X: (..., 24, k) features -> (..., 24) prices."""
        b = self.names.index(BASE)
        return X[..., b] + self.intercept + ((X - self.mean) / self.std) @ self.coef

    def save(self, path=MODEL_FILE):
        np.savez(path, names=np.array(self.names), mean=self.mean, std=self.std,
                 coef=self.coef, intercept=self.intercept,
                 settings=np.array([self.ridge, self.huber_delta, self.n_iter]))

    @classmethod
    def load(cls, path=MODEL_FILE):
        with np.load(path) as d:
            m = cls(d["settings"][0], d["settings"][1], int(d["settings"][2]))
            m.names = d["names"].tolist()
            m.mean, m.std, m.coef = d["mean"], d["std"], d["coef"]
            m.intercept = float(d["intercept"])
        return m


_model = None


def predict_prices(month, day, weekday, net_load_forecast, past_net_load_forecasts,
                   past_realized_net_load, past_realized_da_prices):
    """Predict the trading day's 24 prices from the scheduler's inputs."""
    global _model
    if _model is None:
        _model = PricePredictor.load()
    X, names = build_features(month, day, weekday, net_load_forecast,
                              past_net_load_forecasts, past_realized_net_load,
                              past_realized_da_prices)
    if names != _model.names:
        raise ValueError("price_model.npz was trained on different features; retrain it")
    return _model.predict(X)


def metrics(pred, actual):
    """Daily-forecast metrics. The last two measure hour ordering, not price level.

    rank_corr: mean within-day rank correlation.
    spread_capture: realized (2 highest - 2 lowest predicted hours) / perfect-foresight
    spread. Two hours is the battery's full charge time; hour order is ignored.
    """
    e = pred - actual
    rows = np.arange(len(pred))[:, None]

    def spread(q):
        o = np.argsort(q, axis=1)
        return (actual[rows, o[:, -2:]].sum(1) - actual[rows, o[:, :2]].sum(1)).sum()

    def ranks(q):
        return np.argsort(np.argsort(q, axis=1), axis=1)

    rc = [np.corrcoef(a, b)[0, 1] for a, b in zip(ranks(pred), ranks(actual))]
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e**2).mean()),
            "median_AE": np.median(np.abs(e)), "rank_corr": np.mean(rc),
            "spread_capture": spread(pred) / spread(actual)}


def walk_forward(X, Y, names, first=90, step=15, **kw):
    """Predict days first.. in blocks, each using a model fit only on earlier days."""
    out = []
    for s in range(first, len(X), step):
        out.append(PricePredictor(**kw).fit(X[:s], Y[:s], names).predict(X[s:s + step]))
    return np.concatenate(out)


def main(csv_path):
    X, Y, names, _ = build_dataset(load_days(csv_path))
    PricePredictor().fit(X, Y, names).save()
    print(f"trained on {len(X)} days, {len(names)} features -> {MODEL_FILE.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python price_predictor.py path/to/training_data.csv")
    main(sys.argv[1])
