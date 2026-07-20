# Megaline Plan Classifier

A supervised binary classification model that recommends one of two mobile plans — **Smart** or **Ultra** — from a subscriber's monthly usage behavior. Trained and evaluated on 3,214 user-month records for the fictional carrier Megaline.

**Result: 80.7% accuracy on a held-out test set**, against a required threshold of 0.75 and a majority-class baseline of 69.4%.

This project was completed as the Sprint 8 project in the TripleTen Data Science / ML program. It was **approved on the first submission with no required revisions**.

![tests](https://github.com/N4D3RI/megaline-plan-classifier/actions/workflows/tests.yml/badge.svg)

---

## Project Goal

Megaline wants to migrate subscribers off legacy plans. Given a user's calls, minutes, messages, and data usage for a month, predict which of the two newer plans that user should be on.

The business requirement: **accuracy of at least 0.75 on data the model has never seen.**

## Dataset

Source file: `users_behavior.csv` (provided by TripleTen) — 3,214 rows, 5 columns, one row per user-month.

| Column | Description |
|---|---|
| `calls` | Number of calls |
| `minutes` | Total call duration (minutes) |
| `messages` | Number of text messages |
| `mb_used` | Internet traffic used (MB) |
| `is_ultra` | **Target** — Ultra = 1, Smart = 0 |

The data arrived already cleaned: no missing values, all numeric, no duplicates to resolve. The target is imbalanced — **69.4% Smart / 30.6% Ultra** — which is the single most important fact in the project, because it sets the bar any model has to clear to be worth anything.

> The raw dataset is not redistributed in this repo (it's TripleTen course material). The notebook references it as `/datasets/users_behavior.csv`.

---

## Workflow

**1. Load and inspect** — confirmed 3,214 rows with no missing values, and measured the class balance of the target before doing anything else.

**2. Split into train / validation / test (60 / 20 / 20)** — done in two `train_test_split` steps with `stratify` on both, so the 69/31 class ratio is preserved in every subset. The test set is held out and touched exactly once, at the end.

| Subset | Rows | Purpose |
|---|---|---|
| Train | 1,928 | Fit the models |
| Validation | 643 | Compare models and tune hyperparameters |
| Test | 643 | Single unbiased final estimate |

**3. Investigate models and hyperparameters** — three algorithms compared **on the validation set only**, with a fixed `random_state=12345` throughout for reproducibility:

- Decision Tree — swept `max_depth` from 1 to 10
- Random Forest — grid over `n_estimators` (10–100, step 10) × `max_depth` (1–10), 100 combinations
- Logistic Regression — linear baseline, `liblinear` solver

**4. Evaluate on the test set** — the winning configuration was refit on train + validation combined (80% of the data) so the final model uses as much information as possible, then scored once on the untouched test set.

**5. Sanity check** — compared against a `DummyClassifier(strategy='most_frequent')` to confirm the model learned real signal rather than just exploiting the class imbalance.

---

## Results

### Model comparison (validation set)

| Model | Best hyperparameters | Validation accuracy |
|---|---|---|
| **Random Forest** | `n_estimators=10`, `max_depth=8` | **0.8320** |
| Decision Tree | `max_depth=5` | 0.8165 |
| Logistic Regression | `solver='liblinear'` | 0.7045 |

### Decision Tree depth sweep

| `max_depth` | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Validation accuracy | .7589 | .7838 | .8040 | .8040 | **.8165** | .8025 | .8072 | .8056 | .8118 | .8056 |

Depth 1–2 clearly underfits. From depth 3 onward accuracy sits in a narrow .8025–.8165 band, peaking at depth 5 — the differences between those depths are small enough that they are within noise for a 643-row validation set.

### Final model on the test set

| Metric | Value |
|---|---|
| **Test accuracy** | **0.8072** |
| Required threshold | 0.75 — **met** |
| Majority-class baseline | 0.6936 |
| **Improvement over baseline** | **+0.1135** |

---

## Key Findings

- **Random Forest won**, at 0.8320 validation / 0.8072 test accuracy. Averaging over many trees suppresses the variance that makes a single deep tree unstable.
- **Logistic Regression was the weakest model by a wide margin** (0.7045 — barely above the 0.6936 baseline). The decision boundary between the plans is non-linear, so a linear model has almost nothing to work with here. This is the clearest signal in the project that model choice has to follow the shape of the data.
- **The best forest used the fewest trees in the grid** (`n_estimators=10`). Adding trees beyond that did not improve validation accuracy, so the extra training cost would have bought nothing on this dataset.
- **Depth is the hyperparameter that moves the single decision tree.** The sweep spans nearly 6 points (0.7589 → 0.8165) — shallow trees underfit, and accuracy stops improving once depth passes 5.
- **The sanity check is what makes the headline number meaningful.** On a 69/31 split, a model that predicts "Smart" every single time scores 0.6936. Reporting 0.8072 without that comparison would overstate the model's value; the honest claim is that it adds **11.4 points over doing nothing**.

## Conclusion

The Random Forest meets Megaline's 0.75 requirement with room to spare and clearly beats the majority-class baseline, so it is learning genuine structure in subscriber behavior rather than exploiting the class imbalance. It is fit for use as a plan-recommendation model.

---

## Extended analysis

Accuracy is a weak summary for a 69/31 target: it says nothing about whether the model works for the minority class, which is the one Megaline actually wants to identify. The project reviewer raised the same point, recommending per-class metrics, a confusion matrix, and a stability check.

`megaline.py` and `run_analysis.py` implement that extension:

| Addition | Why it matters here |
|---|---|
| Per-class precision / recall / F1 | Headline accuracy is carried by the 69% majority class; these separate Ultra performance from Smart performance |
| Confusion matrix | Shows *how* the errors are distributed, not just how many |
| ROC-AUC and average precision | Threshold-independent, and average precision is the appropriate summary under class imbalance |
| Seed stability (7 seeds) | Turns a single accuracy figure into a mean and a spread |
| 5-fold stratified cross-validation | A 643-row validation split cannot separate configurations that differ by a point; CV gives a standard deviation |
| Permutation importance | Measured on held-out data, so it avoids the training-set bias of the forest's built-in `feature_importances_` |
| Full grid recorded | The original run kept only the winning configuration, which made claims about hyperparameter sensitivity unsupportable |

Reproduce everything with:

```bash
pip install -r requirements.txt
python run_analysis.py --data /datasets/users_behavior.csv
```

This writes `results/RESULTS.md` (all tables) and seven figures to `figures/`. Every number in that report is generated by the script — none is hand-entered.

## Tests

```bash
pytest test_megaline.py -v
```

16 tests, run against a synthetic frame with the same schema and class balance as the real data, so the suite passes without the non-redistributable course dataset. They assert the properties this write-up claims:

- splits are 60/20/20 and stratification preserves the class ratio in every subset
- the test set is disjoint from train, validation, and the refit pool — the leakage check that makes the headline metric meaningful
- results are reproducible under a fixed seed, and genuinely change under a different one
- metrics match hand-computed values on a known confusion matrix
- `evaluate` returns NaN rather than a silently wrong AUC for models without `predict_proba`
- the majority baseline never predicts the minority class

---

## Stack

Python 3, pandas, NumPy, scikit-learn, Matplotlib, Jupyter. Tested on Python 3.10–3.12 via GitHub Actions.

## Files

| File | Contents |
|---|---|
| `notebook.ipynb` | The approved submission — full analysis, code, plots, conclusions |
| `megaline.py` | Reusable pipeline: loading, splitting, tuning, evaluation, stability |
| `run_analysis.py` | Regenerates every table and figure from the raw dataset |
| `test_megaline.py` | Test suite for the pipeline |
| `requirements.txt` | Dependencies |

## Reproducibility

Every model uses `random_state=12345`, both splits are stratified, and `load_data` fails loudly if the schema or completeness assumptions are violated rather than silently producing wrong numbers downstream.

## Status

Reviewed and **approved on the first submission, with no required revisions**, by TripleTen's review team (July 2026). The extended analysis above implements the reviewer's optional recommendations.
