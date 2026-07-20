"""
Regenerate every table and figure in RESULTS.md from the raw dataset.

    python run_analysis.py --data /datasets/users_behavior.csv

Writes to results/ (tables as CSV + a markdown summary) and figures/ (PNG).
Nothing in the write-up is hand-entered — if a number appears in RESULTS.md it
was produced by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

import megaline as mg

RESULTS = Path("results")
FIGURES = Path("figures")


def _save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(data_path: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    df = mg.load_data(data_path)
    splits = mg.make_splits(df)

    balance = df[mg.TARGET].value_counts(normalize=True).sort_index()
    print(f"rows={len(df)}  smart={balance[0]:.4f}  ultra={balance[1]:.4f}")
    print(f"splits: {splits.sizes}")

    # -- 1. model search ---------------------------------------------------
    tree = mg.tune_decision_tree(splits)
    forest = mg.tune_random_forest(splits)
    logistic = mg.fit_logistic(splits)

    comparison = pd.DataFrame(
        [
            {
                "model": r.name,
                "best_params": str(r.best_params),
                "valid_accuracy": r.best_valid_accuracy,
            }
            for r in (forest, tree, logistic)
        ]
    ).sort_values("valid_accuracy", ascending=False)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
    tree.trials.to_csv(RESULTS / "decision_tree_sweep.csv", index=False)
    forest.trials.to_csv(RESULTS / "random_forest_grid.csv", index=False)

    # How much does n_estimators actually matter? Best accuracy per tree count.
    by_trees = (
        forest.trials.groupby("n_estimators")["valid_accuracy"]
        .max()
        .reset_index()
        .rename(columns={"valid_accuracy": "best_valid_accuracy"})
    )
    by_trees.to_csv(RESULTS / "forest_accuracy_by_n_estimators.csv", index=False)

    # -- 2. final model ----------------------------------------------------
    final = RandomForestClassifier(random_state=mg.RANDOM_STATE, **forest.best_params)
    final.fit(splits.X_pool, splits.y_pool)

    test_metrics = mg.evaluate(final, splits.X_test, splits.y_test)
    baseline_metrics = mg.majority_baseline(splits)

    metrics_table = pd.DataFrame(
        {"final_model": test_metrics, "majority_baseline": baseline_metrics}
    )
    metrics_table.to_csv(RESULTS / "test_metrics.csv")

    # -- 3. stability ------------------------------------------------------
    stability = mg.seed_stability(forest.best_params, splits)
    stability.to_csv(RESULTS / "seed_stability.csv", index=False)

    cv = mg.cross_validate_best(forest.best_params, df)
    cv.to_csv(RESULTS / "cross_validation.csv", index=False)

    importance = mg.permutation_importance_table(final, splits.X_test, splits.y_test)
    importance.to_csv(RESULTS / "permutation_importance.csv", index=False)

    # -- 4. figures --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tree.trials["max_depth"], tree.trials["valid_accuracy"], marker="o")
    ax.axhline(baseline_metrics["accuracy"], ls="--", c="grey",
               label=f"majority baseline ({baseline_metrics['accuracy']:.3f})")
    ax.set(xlabel="max_depth", ylabel="validation accuracy",
           title="Decision tree: accuracy vs depth")
    ax.legend()
    _save(fig, "decision_tree_depth_sweep.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(by_trees["n_estimators"], by_trees["best_valid_accuracy"], marker="o")
    ax.set(xlabel="n_estimators", ylabel="best validation accuracy",
           title="Random forest: does adding trees help?")
    _save(fig, "forest_accuracy_by_n_estimators.png")

    pivot = forest.trials.pivot(index="max_depth", columns="n_estimators",
                                values="valid_accuracy")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set(xticks=range(len(pivot.columns)), yticks=range(len(pivot.index)),
           xlabel="n_estimators", ylabel="max_depth",
           title="Random forest grid: validation accuracy")
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax)
    _save(fig, "forest_grid_heatmap.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        final, splits.X_test, splits.y_test,
        display_labels=["Smart", "Ultra"], cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion matrix (test set)")
    _save(fig, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(final, splits.X_test, splits.y_test, ax=ax)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="chance")
    ax.set_title("ROC curve (test set)")
    ax.legend()
    _save(fig, "roc_curve.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_estimator(final, splits.X_test, splits.y_test, ax=ax)
    ax.axhline(df[mg.TARGET].mean(), ls="--", c="grey",
               label=f"prevalence ({df[mg.TARGET].mean():.3f})")
    ax.set_title("Precision-recall curve (test set)")
    ax.legend()
    _save(fig, "precision_recall_curve.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importance["feature"], importance["importance_mean"],
            xerr=importance["importance_std"])
    ax.set(xlabel="drop in accuracy when shuffled",
           title="Permutation importance (test set)")
    ax.invert_yaxis()
    _save(fig, "permutation_importance.png")

    # -- 5. summary --------------------------------------------------------
    lines = [
        "# Results",
        "",
        "Generated by `run_analysis.py`. Do not edit by hand.",
        "",
        f"- Dataset: {len(df)} rows, "
        f"{balance[1]:.1%} Ultra / {balance[0]:.1%} Smart",
        f"- Splits: {splits.sizes}",
        "",
        "## Model comparison (validation set)",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Test-set metrics",
        "",
        metrics_table.to_markdown(),
        "",
        "## Seed stability (final configuration, varying random_state)",
        "",
        stability.to_markdown(index=False),
        "",
        f"Accuracy across seeds: mean {stability['accuracy'].mean():.4f}, "
        f"sd {stability['accuracy'].std():.4f}, "
        f"range {stability['accuracy'].min():.4f}–{stability['accuracy'].max():.4f}",
        "",
        "## 5-fold cross-validation (full dataset)",
        "",
        cv.to_markdown(index=False),
        "",
        f"Mean {cv['accuracy'].mean():.4f}, sd {cv['accuracy'].std():.4f}",
        "",
        "## Permutation importance (test set)",
        "",
        importance.to_markdown(index=False),
        "",
    ]
    (RESULTS / "RESULTS.md").write_text("\n".join(lines))

    print(f"\nwrote {RESULTS}/RESULTS.md and {len(list(FIGURES.glob('*.png')))} figures")
    print(f"test accuracy {test_metrics['accuracy']:.4f} "
          f"| ultra recall {test_metrics['recall_ultra']:.4f} "
          f"| roc-auc {test_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/datasets/users_behavior.csv")
    main(p.parse_args().data)
