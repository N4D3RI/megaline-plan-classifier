# Megaline Plan Classifier

A supervised binary classification model that recommends one of two mobile plans — **Smart** or **Ultra** — from a subscriber's monthly usage behavior. Trained and evaluated on 3,214 user-month records for the fictional carrier Megaline.

**Result: 80.7% accuracy on a held-out test set**, against a required threshold of 0.75 and a majority-class baseline of 69.4%.

This project was completed as the Sprint 8 project in the TripleTen Data Science / ML program and was reviewed and approved by the program's review team.

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

The depth sweep shows the underfit → optimum → overfit curve clearly:

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

The most useful next steps would be reporting precision/recall per class rather than accuracy alone — since the business cost of misrecommending Ultra is not the same as misrecommending Smart — and cross-validation in place of a single validation split, which would tighten the hyperparameter estimates and show whether the gaps between the top configurations are real or noise.

---

## Stack

Python 3, pandas, scikit-learn (`tree`, `ensemble`, `linear_model`, `dummy`, `model_selection`, `metrics`), Jupyter Notebook.

## Files

- `notebook.ipynb` — full analysis, code, outputs, and conclusions
- `requirements.txt` — pinned versions of the libraries used
- `README.md` — this file

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The notebook expects the dataset at `/datasets/users_behavior.csv`. Update the `pd.read_csv(...)` call in the data-loading cell if your dataset lives somewhere else.

All results are reproducible: every model uses `random_state=12345` and both splits are stratified.

## Status

Reviewed and approved by TripleTen's review team (July 2026).
