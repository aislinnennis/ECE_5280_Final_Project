"""Grid search for the supply-curve and correction settings.

Scores walk-forward error on tuning days 90-268 only.
Usage: python tune.py path/to/training_data.csv
"""
import itertools
import sys

import features
from price_predictor import metrics, walk_forward

FIRST, SPLIT = 90, 269


def main(csv_path):
    days = features.load_days(csv_path)
    results = []
    for hw, lw, rec in itertools.product((0.5, 0.75, 1.0, 1.5), (1500.0, 2500.0, 4000.0),
                                         (0.5, 0.7, 0.85)):
        features.HOUR_WIDTH, features.LOAD_WIDTH, features.RECENCY = hw, lw, rec
        X, Y, names, _ = features.build_dataset(days)
        for ridge, delta in itertools.product((300.0, 1000.0, 3000.0), (5.0, 15.0)):
            pred = walk_forward(X[:SPLIT], Y[:SPLIT], names, first=FIRST,
                                ridge=ridge, huber_delta=delta)
            results.append((metrics(pred, Y[FIRST:SPLIT]), (hw, lw, rec, ridge, delta)))
    results.sort(key=lambda r: r[0]["MAE"])
    print("hour_width load_width recency ridge delta |    MAE   RMSE  rank  spread")
    chosen = (0.75, 2500.0, 0.7, 1000.0, 5.0)
    for n, (m, s) in enumerate(results):
        if n < 10 or s == chosen:
            mark = f"  <- chosen, rank {n + 1} of {len(results)}" if s == chosen else ""
            print("{:10} {:10} {:7} {:5.0f} {:5.0f}".format(*s),
                  "| {MAE:6.3f} {RMSE:6.2f} {rank_corr:.3f}  {spread_capture:.3f}".format(**m) + mark)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tune.py path/to/training_data.csv")
    main(sys.argv[1])
