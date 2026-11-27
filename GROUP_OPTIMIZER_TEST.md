# Group optimizer test (ECE5280_Optimizer_v2 notebook)

**Result: the group optimizer does not change which model is best.** At its default
settings (no robustness radius, no continuation value) it returns the same schedules
as our optimizer.py, so every revenue number is the same to within $2.
Turning on its robust radius or continuation value lowers revenue for every model.

The notebook imports `optimizer.py`, which was not shared, so we rebuilt it (`group_optimizer.py`, not in this folder) from the formulas written in the notebook: lossless LP, the exact
state-of-charge LP for the lossy battery with an exact MILP fallback when the
objective is not convex (negative forecast prices), the price radius rho ($/MWh),
and the continuation value v ($/MWh, non-final days only). It reproduces the
notebook's printed custom-price example ($466.91 forecast revenue, $441.87 worst case).

Full table: `results/group_optimizer_comparison.csv`.

## Revenue, same predicted prices for every optimizer

Quarterly CV (Jan 8 - Dec 31, out of sample)

| Model | Battery | Existing MILP/LP | Group rho=0 v=0 | rho=2.5 | rho=10 | v=20 | v=40 |
|---|---|---|---|---|---|---|---|
| 3-feature default | lossless | **$118,265** | $118,265 | $115,978 | $107,652 | $118,265 | $116,931 |
| 7f ridge 1000 | lossless | $117,882 | $117,882 | $116,107 | $106,982 | $117,882 | $116,168 |
| 7f fine-tuned (shared 7-feature model) | lossless | $117,846 | $117,846 | $116,132 | $106,923 | $117,846 | $116,117 |
| 3-feature default | lossy | **$106,403** | $106,402 | $104,286 | $100,949 | $106,402 | $104,759 |
| 7f ridge 1000 | lossy | $106,176 | $106,176 | $104,133 | $100,611 | $106,176 | $103,929 |
| 7f fine-tuned | lossy | $106,183 | $106,182 | $104,119 | $100,534 | $106,182 | $103,919 |

Walk-forward (Mar 1 - Dec 31, trained only on earlier months)

| Model | Battery | Existing | Group rho=0 v=0 | rho=2.5 | rho=10 | v=20 | v=40 |
|---|---|---|---|---|---|---|---|
| 3-feature default | lossless | $99,154 | $99,154 | $97,308 | $91,491 | $99,154 | $98,572 |
| 7f ridge 1000 | lossless | $99,236 | $99,236 | $97,628 | $91,011 | $99,236 | $98,451 |
| 7f fine-tuned | lossless | $99,047 | $99,047 | $97,657 | $90,908 | $99,047 | $98,265 |
| 3-feature default | lossy | $90,076 | $90,076 | $88,323 | $86,539 | $90,076 | $89,338 |
| 7f ridge 1000 | lossy | $90,102 | $90,102 | $88,187 | $86,264 | $90,102 | $89,153 |
| 7f fine-tuned | lossy | $89,809 | $89,809 | $88,367 | $86,151 | $89,809 | $88,779 |

## What this means

- **Why rho=0, v=0 matches exactly:** both optimizers solve the same problem exactly
  (our MILP and their SOC-LP/MILP hybrid both find the true optimum for the forecast).
  On the lossy battery their hybrid used the fast LP on 76% of days and fell back to
  the MILP on the rest (days with negative forecast prices). Differences of $1-2 are
  solver ties.
- **Robust radius hurts:** rho=2.5 costs about $1,500-2,300 and rho=10 about $3,500-11,000,
  for every model, all 6 to 22 standard errors. It makes the battery trade less, but our
  forecast errors are much larger than the radius, so it just gives up profitable trades.
- **Continuation value:** v=20 never changes a schedule (below typical prices);
  v=40 loses $600-2,500. Selling everything each day (our "zero" rule) stays best,
  consistent with the notebook's own backtest.
- **Model ranking is unchanged:** 3-feature default is best under quarterly CV
  (7-feature is -$383 lossless, about -2 SE; -$220 lossy, about -1.3 SE). Under walk-forward
  7f ridge 1000 is ahead by $82 / $26, well within noise (0.5 / 0.2 SE). Same picture as before.
