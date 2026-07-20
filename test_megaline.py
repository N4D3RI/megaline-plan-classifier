"""
Tests for the Megaline pipeline.

These check the properties the write-up actually claims — correct split
proportions, preserved class balance, no train/test leakage, reproducibility,
and metrics that match hand-computed values on a known confusion matrix.

They run on a synthetic frame with the same schema as the real data, so the
suite is runnable without the (non-redistributable) course dataset.
"""

import numpy as np
import pandas as pd
import pytest

import megaline as mg


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """3,214 rows with a ~69/31 target split, matching the real data's shape."""
    rng = np.random.default_rng(0)
    n = 3214
    n_ultra = 985

    y = np.zeros(n, dtype=int)
    y[:n_ultra] = 1
    rng.shuffle(y)

    # Ultra users skew heavier on every feature, so the task is learnable.
    shift = y * 1.4
    return pd.DataFrame(
        {
            "calls": rng.normal(63 + 12 * shift, 20, n).clip(0).round(),
            "minutes": rng.normal(438 + 90 * shift, 150, n).clip(0),
            "messages": rng.normal(38 + 15 * shift, 25, n).clip(0).round(),
            "mb_used": rng.normal(17207 + 3000 * shift, 6000, n).clip(0),
            "is_ultra": y,
        }
    )


# --- splitting ------------------------------------------------------------


def test_split_proportions_are_60_20_20(synthetic_df):
    s = mg.make_splits(synthetic_df)
    total = len(synthetic_df)

    assert s.sizes["train"] == pytest.approx(0.6 * total, abs=2)
    assert s.sizes["valid"] == pytest.approx(0.2 * total, abs=2)
    assert s.sizes["test"] == pytest.approx(0.2 * total, abs=2)
    assert sum(s.sizes.values()) == total


def test_stratification_preserves_class_balance(synthetic_df):
    s = mg.make_splits(synthetic_df)
    overall = synthetic_df[mg.TARGET].mean()

    for y in (s.y_train, s.y_valid, s.y_test):
        assert y.mean() == pytest.approx(overall, abs=0.01)


def test_no_leakage_between_test_and_training_data(synthetic_df):
    """The headline metric is only meaningful if these sets are disjoint."""
    s = mg.make_splits(synthetic_df)

    assert set(s.X_test.index).isdisjoint(s.X_train.index)
    assert set(s.X_test.index).isdisjoint(s.X_valid.index)
    assert set(s.X_test.index).isdisjoint(s.X_pool.index)
    assert set(s.X_pool.index) == set(s.X_train.index) | set(s.X_valid.index)


def test_splits_are_reproducible(synthetic_df):
    a = mg.make_splits(synthetic_df)
    b = mg.make_splits(synthetic_df)
    assert list(a.X_test.index) == list(b.X_test.index)


def test_different_seed_gives_different_split(synthetic_df):
    a = mg.make_splits(synthetic_df, random_state=1)
    b = mg.make_splits(synthetic_df, random_state=2)
    assert list(a.X_test.index) != list(b.X_test.index)


# --- schema validation ----------------------------------------------------


def test_load_data_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"calls": [1.0], "is_ultra": [0]}).to_csv(bad, index=False)

    with pytest.raises(ValueError, match="missing expected columns"):
        mg.load_data(str(bad))


def test_load_data_rejects_missing_values(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "calls": [1.0, np.nan],
            "minutes": [1.0, 2.0],
            "messages": [1.0, 2.0],
            "mb_used": [1.0, 2.0],
            "is_ultra": [0, 1],
        }
    ).to_csv(bad, index=False)

    with pytest.raises(ValueError, match="missing values"):
        mg.load_data(str(bad))


# --- metrics --------------------------------------------------------------


class _FixedPredictor:
    """Returns a canned prediction vector, so metrics can be checked by hand."""

    def __init__(self, preds):
        self._preds = np.asarray(preds)

    def predict(self, X):
        return self._preds


def test_evaluate_matches_hand_computed_confusion_matrix():
    # 10 rows: 6 Smart (0), 4 Ultra (1)
    y_true = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    # Ultra: tp=3, fp=2, fn=1  -> precision 3/5, recall 3/4
    # Smart: tp=4, fp=1, fn=2  -> precision 4/5, recall 4/6

    m = mg.evaluate(_FixedPredictor(y_pred), pd.DataFrame(index=range(10)), y_true)

    assert m["true_positives"] == 3
    assert m["false_positives"] == 2
    assert m["false_negatives"] == 1
    assert m["true_negatives"] == 4

    assert m["precision_ultra"] == pytest.approx(0.6)
    assert m["recall_ultra"] == pytest.approx(0.75)
    assert m["precision_smart"] == pytest.approx(0.8)
    assert m["recall_smart"] == pytest.approx(4 / 6)
    assert m["accuracy"] == pytest.approx(0.7)


def test_evaluate_returns_nan_auc_without_predict_proba():
    """Better a visible NaN than a quietly wrong AUC."""
    y_true = pd.Series([0, 1, 0, 1])
    m = mg.evaluate(_FixedPredictor([0, 1, 0, 1]), pd.DataFrame(index=range(4)), y_true)
    assert np.isnan(m["roc_auc"])


def test_majority_baseline_never_predicts_minority_class(synthetic_df):
    """This is the number the model must beat to be worth anything."""
    s = mg.make_splits(synthetic_df)
    m = mg.majority_baseline(s)

    assert m["true_positives"] == 0
    assert m["recall_ultra"] == 0.0
    assert m["accuracy"] == pytest.approx(1 - synthetic_df[mg.TARGET].mean(), abs=0.02)


# --- search ---------------------------------------------------------------


def test_decision_tree_search_records_every_depth(synthetic_df):
    s = mg.make_splits(synthetic_df)
    r = mg.tune_decision_tree(s, depths=range(1, 6))

    assert len(r.trials) == 5
    assert r.best_valid_accuracy == r.trials["valid_accuracy"].max()
    assert r.best_params["max_depth"] in list(range(1, 6))


def test_random_forest_search_records_full_grid(synthetic_df):
    """The original run kept only the winner; the grid is what supports claims."""
    s = mg.make_splits(synthetic_df)
    r = mg.tune_random_forest(s, n_estimators_grid=(10, 20), depth_grid=(1, 2, 3))

    assert len(r.trials) == 6
    assert set(r.trials["n_estimators"]) == {10, 20}
    assert r.best_valid_accuracy == r.trials["valid_accuracy"].max()


def test_search_beats_baseline_on_learnable_data(synthetic_df):
    s = mg.make_splits(synthetic_df)
    r = mg.tune_random_forest(s, n_estimators_grid=(20,), depth_grid=(3, 5))
    assert r.best_valid_accuracy > 1 - synthetic_df[mg.TARGET].mean()


# --- stability ------------------------------------------------------------


def test_seed_stability_reports_one_row_per_seed(synthetic_df):
    s = mg.make_splits(synthetic_df)
    out = mg.seed_stability({"n_estimators": 20, "max_depth": 5}, s, seeds=(0, 1, 2))

    assert len(out) == 3
    assert out["accuracy"].between(0, 1).all()


def test_cross_validation_returns_one_row_per_fold(synthetic_df):
    out = mg.cross_validate_best({"n_estimators": 20, "max_depth": 5}, synthetic_df, n_splits=4)

    assert len(out) == 4
    assert out["accuracy"].between(0, 1).all()


def test_permutation_importance_ranks_all_features(synthetic_df):
    s = mg.make_splits(synthetic_df)
    r = mg.tune_random_forest(s, n_estimators_grid=(20,), depth_grid=(5,))
    imp = mg.permutation_importance_table(r.estimator, s.X_test, s.y_test, n_repeats=5)

    assert set(imp["feature"]) == set(mg.FEATURES)
    assert imp["importance_mean"].is_monotonic_decreasing
