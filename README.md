# Megaline Plan Classifier

Megaline is a mobile carrier with a lot of subscribers still sitting on legacy plans. This project builds a classifier that looks at one month of someone's usage, calls, minutes, texts and data, and works out whether they belong on the Smart plan or the Ultra plan.

The brief asked for at least 75% accuracy on data the model had never seen. The final random forest gets 80.7%.

That number needs context though, and the context is the interesting part. About 69% of subscribers are on Smart, so a model that ignores the data entirely and guesses "Smart" every single time already scores 69.4%. The real result is not 80.7%. It is the 11.4 points of separation between those two figures.

![tests](https://github.com/N4D3RI/megaline-plan-classifier/actions/workflows/tests.yml/badge.svg)

Completed as the Sprint 8 project in the TripleTen Data Science and ML program, and approved on the first submission with no required revisions.

## The data

3,214 rows, one per subscriber month, from `users_behavior.csv`.

| Column | What it is |
|---|---|
| `calls` | Number of calls |
| `minutes` | Total call duration |
| `messages` | Number of texts |
| `mb_used` | Data used, in MB |
| `is_ultra` | Target. Ultra is 1, Smart is 0 |

It arrived clean. No missing values, everything numeric, nothing to repair. So the only thing worth noting before modelling is the class balance: 69.4% Smart against 30.6% Ultra. That imbalance shapes every decision that follows.

The raw file is TripleTen course material so it is not redistributed here. The notebook reads it from `/datasets/users_behavior.csv`.

## How I approached it

I split the data 60/20/20 into training, validation and test, using two calls to `train_test_split` with stratification on both, so the 69/31 ratio survives into all three subsets. The test set gets touched exactly once, at the very end.

| Subset | Rows | Used for |
|---|---|---|
| Train | 1,928 | Fitting models |
| Validation | 643 | Comparing models and tuning |
| Test | 643 | One final unbiased estimate |

Then I compared three algorithms, tuning each against the validation set only, with `random_state=12345` fixed throughout:

* a decision tree, sweeping `max_depth` from 1 to 10
* a random forest, over a 100 point grid of `n_estimators` from 10 to 100 crossed with `max_depth` from 1 to 10
* logistic regression as a linear reference point

The winner was refit on training plus validation combined, so the final model uses 80% of the data, then scored once against the test set. Finally I checked it against a `DummyClassifier` that always predicts the majority class, to confirm it had learned something real rather than just absorbing the imbalance.

## What happened

Validation accuracy:

| Model | Best settings | Accuracy |
|---|---|---|
| Random forest | `n_estimators=10`, `max_depth=8` | 0.8320 |
| Decision tree | `max_depth=5` | 0.8165 |
| Logistic regression | `solver='liblinear'` | 0.7045 |

The decision tree depth sweep:

| `max_depth` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | .7589 | .7838 | .8040 | .8040 | .8165 | .8025 | .8072 | .8056 | .8118 | .8056 |

Depth 1 and 2 clearly underfit. After that it flattens out into a narrow band between .8025 and .8165, peaking at 5. I want to be careful not to over read that peak: on 643 validation rows, gaps that small are comfortably inside the noise.

On the test set:

| | Value |
|---|---|
| Test accuracy | 0.8072 |
| Required | 0.75, cleared |
| Majority class baseline | 0.6936 |
| Gain over baseline | 0.1135 |

## What I took from it

Logistic regression was the most informative failure. At 0.7045 it barely clears the do nothing baseline of 0.6936, which says the boundary between the two plans is not linear. Once you see that, the tree based models winning is not a surprise, it is the expected consequence.

The best forest turned out to use only 10 trees, the smallest value in the grid. Adding more bought nothing here. Worth remembering next time I am tempted to reach for a bigger ensemble by reflex.

For the single tree, depth is the lever that matters. It moves accuracy by almost 6 points across the sweep, from .7589 up to .8165, while nothing else comes close.

And the baseline comparison is the thing I would want someone to check first. Reporting 80.7% on its own would flatter the model. Against a 69/31 split the honest claim is 11.4 points better than guessing.

## Going further than the brief required

Accuracy is a poor summary when the classes are this lopsided. It tells you nothing about whether the model actually finds Ultra users, which is the group Megaline cares about. The project reviewer flagged the same gap and suggested per class metrics, a confusion matrix, and a stability check.

`megaline.py` and `run_analysis.py` build that out:

| Added | Why |
|---|---|
| Precision, recall and F1 per class | Splits Ultra performance out from Smart, instead of letting the 69% majority carry the headline |
| Confusion matrix | Shows how the errors distribute, not just how many there are |
| ROC AUC and average precision | Independent of the 0.5 threshold, and average precision is the right summary under imbalance |
| Seed stability across 7 seeds | Turns one accuracy figure into a mean and a spread |
| 5 fold stratified cross validation | 643 validation rows cannot separate models a point apart, so this gives a standard deviation instead |
| Permutation importance | Measured on held out rows, avoiding the training set bias in the forest's own `feature_importances_` |
| The full grid, saved | The first version kept only the winner, which left me unable to say anything about how the search behaved |

To regenerate everything:

```bash
pip install -r requirements.txt
python run_analysis.py --data /datasets/users_behavior.csv
```

That writes `results/RESULTS.md` and seven figures into `figures/`. Every number in that report comes out of the script. None of it is typed in by hand.

## Tests

```bash
pytest test_megaline.py -v
```

18 tests, running against a synthetic frame built to match the real data's schema and class balance, so the suite works without the course dataset. They check the claims this README makes rather than just exercising the code:

* splits come out 60/20/20 and stratification holds in every subset
* the test set is disjoint from train, validation and the refit pool, which is the leakage check the headline number depends on
* a fixed seed reproduces exactly, and a different seed genuinely changes the split
* metrics match values worked out by hand from a known confusion matrix
* `evaluate` returns NaN rather than a wrong AUC when a model has no `predict_proba`
* the majority baseline never once predicts the minority class
* parallelism is a speed setting only, and does not shift results

## Stack

Python 3, pandas, NumPy, scikit-learn, Matplotlib, Jupyter. CI runs the suite on Python 3.10, 3.11 and 3.12.

## Files

| File | Contents |
|---|---|
| `notebook.ipynb` | The submitted analysis, with code, plots and conclusions |
| `megaline.py` | The pipeline as reusable functions |
| `run_analysis.py` | Regenerates every table and figure from the raw data |
| `test_megaline.py` | Test suite |
| `requirements.txt` | Dependencies |

## Reproducibility

Every model uses `random_state=12345`, both splits are stratified, and `load_data` raises immediately if the schema or completeness assumptions break, rather than letting bad numbers flow downstream unnoticed.
