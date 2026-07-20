"""
Megaline plan classifier — reusable analysis pipeline.

Every function is deterministic given `random_state`. Nothing here reads or
writes global state, so each step can be tested in isolation.

The public surface is small on purpose:

    load_data          -> DataFrame
    make_splits        -> Splits (train / valid / test, stratified)
    tune_decision_tree -> SearchResult
    tune_random_forest -> SearchResult
    fit_logistic       -> SearchResult
    evaluate           -> dict of test-set metrics
    seed_stability     -> DataFrame, one row per seed
    cross_validate_best-> DataFrame, one row per fold
    permutation_importance_table -> DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 12345
FEATURES = ["calls", "minutes", "messages", "mb_used"]
TARGET = "is_ultra"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_data(path: str = "/datasets/users_behavior.csv") -> pd.DataFrame:
    """Load the behaviour table and assert the schema we expect downstream."""
    df = pd.read_csv(path)

    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing expected columns: {sorted(missing)}")

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("dataset contains missing values; clean before modelling")

    return df


@dataclass(frozen=True)
class Splits:
    """Train / validation / test partition, plus the pooled train+valid set.

    `X_pool` / `y_pool` exist because the final model is refit on train+valid
    once hyperparameters are chosen — keeping it here means the refit can never
    accidentally include test rows.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_pool: pd.DataFrame
    y_pool: pd.Series

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.X_train),
            "valid": len(self.X_valid),
            "test": len(self.X_test),
        }


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    valid_size_of_pool: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> Splits:
    """Stratified 60/20/20 split, done in two steps.

    Step 1 holds out the test set. Step 2 splits the remainder, so the test set
    is never touched during tuning. Both steps stratify on the target to keep
    the class ratio stable across all three subsets.
    """
    X = df[FEATURES]
    y = df[TARGET]

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_pool,
        y_pool,
        test_size=valid_size_of_pool,
        random_state=random_state,
        stratify=y_pool,
    )

    return Splits(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        X_pool=X_pool,
        y_pool=y_pool,
    )


# --------------------------------------------------------------------------
# Hyperparameter search
# --------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Outcome of one model's hyperparameter search.

    `trials` keeps every configuration tried, not just the winner — without it
    you cannot tell whether the best score is a real peak or a lucky draw.
    """

    name: str
    best_params: dict[str, Any]
    best_valid_accuracy: float
    trials: pd.DataFrame
    estimator: Any


def tune_decision_tree(
    splits: Splits,
    depths: Iterable[int] = range(1, 11),
    random_state: int = RANDOM_STATE,
) -> SearchResult:
    rows = []
    best = (-1.0, None, None)

    for depth in depths:
        model = DecisionTreeClassifier(random_state=random_state, max_depth=depth)
        model.fit(splits.X_train, splits.y_train)
        acc = accuracy_score(splits.y_valid, model.predict(splits.X_valid))
        rows.append({"max_depth": depth, "valid_accuracy": acc})
        if acc > best[0]:
            best = (acc, {"max_depth": depth}, model)

    return SearchResult(
        name="Decision Tree",
        best_params=best[1],
        best_valid_accuracy=best[0],
        trials=pd.DataFrame(rows),
        estimator=best[2],
    )


def tune_random_forest(
    splits: Splits,
    n_estimators_grid: Iterable[int] = range(10, 101, 10),
    depth_grid: Iterable[int] = range(1, 11),
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> SearchResult:
    """Full grid, recording every cell.

    The first version of this project kept only the winning configuration,
    which made it impossible to say afterwards how much `n_estimators` actually
    mattered. Keeping the whole grid is what lets the write up say anything
    about the shape of the search rather than just its winner.

    `n_jobs` only parallelises tree building inside each fit, so results stay
    identical for a given `random_state`.
    """
    rows = []
    best = (-1.0, None, None)

    for n_estimators in n_estimators_grid:
        for depth in depth_grid:
            model = RandomForestClassifier(
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=depth,
                n_jobs=n_jobs,
            )
            model.fit(splits.X_train, splits.y_train)
            acc = accuracy_score(splits.y_valid, model.predict(splits.X_valid))
            rows.append(
                {
                    "n_estimators": n_estimators,
                    "max_depth": depth,
                    "valid_accuracy": acc,
                }
            )
            if acc > best[0]:
                best = (
                    acc,
                    {"n_estimators": n_estimators, "max_depth": depth},
                    model,
                )

    return SearchResult(
        name="Random Forest",
        best_params=best[1],
        best_valid_accuracy=best[0],
        trials=pd.DataFrame(rows),
        estimator=best[2],
    )


def fit_logistic(
    splits: Splits, random_state: int = RANDOM_STATE
) -> SearchResult:
    model = LogisticRegression(random_state=random_state, solver="liblinear")
    model.fit(splits.X_train, splits.y_train)
    acc = accuracy_score(splits.y_valid, model.predict(splits.X_valid))

    return SearchResult(
        name="Logistic Regression",
        best_params={"solver": "liblinear"},
        best_valid_accuracy=acc,
        trials=pd.DataFrame([{"solver": "liblinear", "valid_accuracy": acc}]),
        estimator=model,
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate(model, X, y_true) -> dict[str, Any]:
    """Metrics for one fitted model on one set of rows.

    Accuracy alone is misleading on a 69/31 split, so this also reports
    precision, recall and F1 for each class separately, plus two scores that
    do not depend on the 0.5 decision threshold. ROC AUC and average precision
    need predicted probabilities, so a model without `predict_proba` gets NaN
    for those rather than a quietly wrong number.

    Returns floats for the rates and ints for the four confusion matrix cells.
    """
    y_pred = model.predict(X)

    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_ultra": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_ultra": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_ultra": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_smart": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_smart": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_smart": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_true, proba)
        metrics["average_precision"] = average_precision_score(y_true, proba)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
    )

    return metrics


def majority_baseline(splits: Splits) -> dict[str, float]:
    """The number every other result has to be read against."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(splits.X_train, splits.y_train)
    return evaluate(dummy, splits.X_test, splits.y_test)


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------


def seed_stability(
    best_params: dict[str, Any],
    splits: Splits,
    seeds: Iterable[int] = (0, 1, 7, 42, 123, 2024, 12345),
) -> pd.DataFrame:
    """Refit the winning configuration under several seeds.

    A single accuracy figure hides how much of the result is the seed. The
    spread across seeds is the honest error bar on the headline number.
    """
    rows = []
    for seed in seeds:
        model = RandomForestClassifier(random_state=seed, **best_params)
        model.fit(splits.X_pool, splits.y_pool)
        m = evaluate(model, splits.X_test, splits.y_test)
        rows.append({"seed": seed, **{k: m[k] for k in
                                      ("accuracy", "f1_ultra", "recall_ultra", "roc_auc")}})
    return pd.DataFrame(rows)


def cross_validate_best(
    best_params: dict[str, Any],
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Stratified k-fold on the full dataset.

    A single 20% validation split is ~643 rows, so differences of a point or
    two between configurations sit inside the noise. Cross-validation gives a
    mean and a standard deviation instead of one draw.
    """
    model = RandomForestClassifier(random_state=random_state, **best_params)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, df[FEATURES], df[TARGET], cv=cv, scoring="accuracy")
    return pd.DataFrame({"fold": range(1, n_splits + 1), "accuracy": scores})


def permutation_importance_table(
    model, X, y, n_repeats: int = 20, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Permutation importance, measured on held-out data.

    Preferred over the forest's built-in `feature_importances_`, which is
    computed on the training set and is biased toward high-cardinality
    features.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring="accuracy"
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
