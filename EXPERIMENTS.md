# Experiments and results

All numbers are dollars of revenue from our local copy of the Gradescope evaluator, which replays the
training data day by day (battery starts full, energy carries over, Jan 8 to Dec 31 unless noted).
"Perfect" means perfect foresight of prices: $123,045 lossless and $113,867 lossy.

## How we measured (no peeking)

* **Quarterly cross-validation.** For each quarter, train on the other three quarters and trade the held-out
  quarter. Training days in the week right after the test quarter are dropped, because their "past prices"
  inputs contain test-quarter prices. All four quarters together give one out-of-sample score for the year.
* **Noise check.** For two setups we compare revenue day by day and compute the standard error (SE) of the total
  difference. A gap under about 2 SE is treated as noise (`paired_check.py`).
* **Features use only the 9 scheduler inputs**, through the same function at training and run time.
* **Data cleaning.** Jan 11, hours 15-23 of realized net load are a straight-line fill; we treat them as missing.
  Price spikes and negative prices are real and kept.
* **End of day.** Emptying the battery at the end of every day beat valuing leftover energy at the mean price
  ($116,737 vs $115,621), so the schedule empties daily.

Baseline: "tomorrow's prices = yesterday's prices" earns $115,628 lossless (94.0% of perfect).
So any model gains only a few thousand dollars on top of that.

## 1. Adding features one at a time (`compare_features.py`)

Start from hour of day and add features in a fixed order.

| features | RMSE ($/MWh) | lossless revenue | % of perfect |
|---|---|---|---|
| hour | 30.30 | $107,909 | 87.7% |
| + net-load forecast | 22.94 | $116,737 | 94.9% |
| + yesterday's price | 16.00 | $117,759 | 95.7% |
| + all 11 original features | 14.71 | $117,324 | 95.4% |

![Revenue as features are added, lossless](results/features_lossless.png)
![Revenue as features are added, lossy](results/features_lossy.png)

**Finding:** lower prediction error does not always mean more revenue. Adding "forecast change" cut RMSE from 16.0
to 14.6 but revenue went down slightly. The schedule only cares which hours are cheap or expensive, not the exact price.

## 2. Comparing model types (`compare_models.py`)

Same 3 features (hour, net-load forecast, yesterday's price), lossless:

| model | revenue | RMSE |
|---|---|---|
| least squares | $117,747 | 16.00 |
| ridge, penalty 1 / 10 / 100 / 1000 | $117,759 / $117,735 / $117,849 / $117,956 | 16.0 to 16.3 |
| ridge, penalty 10000 | $117,349 | 21.93 |
| one ridge per hour (penalty 1 / 100) | $114,481 / $116,596 | 16.1 / 16.6 |
| lasso 2.0 | $118,001 | 16.08 |
| gradient-boosted trees | $111,183 | 17.76 |

![Model comparison, lossless](results/models_lossless.png)
![Model comparison, lossy](results/models_lossy.png)

**Finding:** all pooled linear models are within about $250 (noise). Per-hour models and trees are clearly worse:
with one year of data they overfit. Lossy shows the same pattern.

## 3. Lasso: which features survive (`lasso_features.py`)

All 11 features, increasing penalty. Dropped first: 7-day average price, then forecast shape, yesterday's forecast
error, hour, weekday, then last week's price and season. Last ones standing: yesterday's price, net-load forecast,
forecast change. Weights: `results/lasso_weights.csv`.

**Finding:** lasso drops hour because yesterday's price at the same hour already carries the daily shape. It keeps
forecast change, which helps prediction error but not revenue: lasso picks features for accuracy, not for scheduling.

## 4. Ranking features by revenue: drop-one and backward elimination (`feature_ranking.py`)

Start from all 11 features and remove one at a time; measure the revenue lost.

![Revenue lost when each feature is removed, lossless](results/drop_one_lossless.png)
![Revenue lost when each feature is removed, lossy](results/drop_one_lossy.png)

Lossless: removing yesterday's price costs $406, net-load forecast $166; removing most others costs nothing or helps.
Backward elimination (keep deleting the least useful feature) ends at **net-load forecast + yesterday's price**:
lossless $117,949 (95.9%), lossy $106,085.

## 5. Splitting the model into pieces (piecewise ridge, `compare_features.py --split`)

With the 2 best features, lossless: one pooled ridge $117,949; four 6-hour blocks $116,338 to $117,044;
three load levels $115,762 to $118,049; one per hour $114,481 to $116,596.
Table: `results/models_lossless_ridge.csv`.

**Finding:** splitting only matches the pooled model when the penalty is so large that it pushes the pieces back
together. Not worth it with one year of data.

## 6. Ramp, season and spike features

Added one at a time to net-load forecast + yesterday's price:

| feature | lossless | lossy |
|---|---|---|
| base | $117,949 | $106,085 |
| + past ramp 1h / 2h / 3h | +$235 / **+$316** / +$176 | +$276 / **+$318** / +$203 |
| + next ramp 1h / 2h / 3h | -$136 / -$216 / -$285 | -$71 / -$342 / -$477 |
| + season / season x load / summer x load | +$22 / +$66 / +$46 | +$44 / +$91 / +$74 |
| + load squared / load above last week's peak | -$204 / -$59 | -$123 / -$82 |

(Ramp = forecast now minus forecast k hours earlier.) The 2-hour past ramp's gain is real (3.7 SE, lossy); the
season features are noise (about 0.6 SE). This gives our **default 3-feature model**:
net-load forecast + yesterday's price + 2-hour ramp, **lossless $118,265 (96.1%), lossy $106,403**.

**Finding (why season does not help):** a lossless schedule depends only on how the 24 hours of a day compare with
each other. A feature that is the same for every hour of the day (season, month, weekday) shifts all 24 predictions
up or down together, so the cheapest and most expensive hours stay the same and the schedule does not change.

## 7. Forward stepwise search (`stepwise_search.py`)

From 12 candidate features, add the one that raises out-of-sample revenue most, repeat, then a backward pass.

![Stepwise search, lossless](results/stepwise_lossless.png)
![Stepwise search, lossy](results/stepwise_lossy.png)

Best set: 7-day average price, net-load forecast, 2-hour ramp, load squared: lossless $118,421, only +$156 over the
default (0.9 SE, noise). Revenue peaks at 3-5 features and falls as more are added (all 12: $116,911), a sign of
overfitting. We kept the smaller 3-feature default.

## 8. The 7-feature model with revenue fine-tuning (`compare_alt_system.py`, `decision_tuning.py`)

Features: hour, weekday, month, net-load forecast, forecast change, yesterday's price, 7-day average price.
Ridge penalty 1000, then the weights are fine-tuned to maximize lossless revenue on the training days
(random hill climb, 150 tries, seed 0).

| system | lossless | lossy |
|---|---|---|
| default (3 features, ridge 1) | $118,265 | $106,403 |
| 7 features, ridge 1000 | $117,882 (-$383, 2.1 SE) | $106,176 (-$227, 1.3 SE) |
| 7 features, ridge 1000 + fine-tuned | $117,846 (-$419, 2.3 SE) | $106,183 (-$220, 1.3 SE) |

**Finding:** fine-tuning raised training revenue in every fold ($54 to $351) but did not carry over to held-out
data; it fits the training days' noise.

## 9. Walk-forward vs quarterly cross-validation (`walk_forward.py`)

Walk-forward: for each month Mar-Dec, train on all earlier days and trade that month. Both methods scored on the same
306 days (Mar 1 - Dec 31). In brackets: difference from the default in SE.

| setup | lossless WF | lossless CV | lossy WF | lossy CV |
|---|---|---|---|---|
| naive: today = yesterday | $97,190 (-8.9) | $97,190 (-9.3) | $88,265 (-7.2) | $88,265 (-7.5) |
| default (3 features) | $99,154 | $99,312 | $90,076 | $90,209 |
| stepwise winner (4 features) | $98,910 (-0.6) | $99,306 (0.0) | $89,857 (-0.6) | $90,289 (+0.5) |
| 7 features, ridge 1000 | $99,236 (+0.5) | $99,120 (-1.2) | $90,102 (+0.2) | $90,088 (-0.8) |

**Finding:** every model beats the naive baseline by a wide margin under both methods. Among the three models the
ranking changes between methods but all gaps are noise. The stepwise winner's small edge disappears under walk-forward,
so it was partly fitted to the CV numbers.

## 10. Testing the group's optimizer (ECE5280_Optimizer_v2 notebook)

Same predicted prices, our optimizer vs the group optimizer. At its defaults (robust radius rho = 0,
continuation value v = 0) it gives the same schedules and the same revenue to within $2. A robust radius
lowers revenue for every model (rho = 2.5: about -$1,500 to -$2,300; rho = 10: -$3,500 to -$11,000), and
v = 40 loses $600 to $2,500. The model ranking does not change. Details: [GROUP_OPTIMIZER_TEST.md](GROUP_OPTIMIZER_TEST.md),
table `results/group_optimizer_comparison.csv`.

## Conclusion

The 3-feature ridge model (net-load forecast, yesterday's price, 2-hour ramp) is as good as anything else we tried
and is the simplest: about 96% of perfect-foresight revenue lossless and 93% lossy, out of sample.
The 4-feature stepwise set and the 7-feature model are reasonable candidates to compare on the Gradescope
validation evaluator, which is the best tie-breaker.

All result tables are in `results/*.csv`. The script names in brackets refer to our analysis code, which is not in this folder.
