from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate

from custom.ensemble import ExtraTreesClassifier as CustomExtraTreesClassifier
from custom.ensemble import RandomForestClassifier as CustomRandomForestClassifier
from custom.tree import DecisionTreeClassifier as CustomDecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier as SklearnExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

from utils.analysis import (
    BiasVarianceAnalyzer,
    DatasetAnalyzer,
    ModelPerformanceAnalyzer,
)
from utils.dataset import load_and_split
from utils.evaluate import compute_classification_metrics


@dataclass(frozen=True)
class MetricSummary:
    values: list[float]
    mean: float
    std: float


def summarize_metric(values: list[float]) -> MetricSummary:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return MetricSummary(values=[], mean=0.0, std=0.0)
    return MetricSummary(
        values=array.tolist(),
        mean=float(np.mean(array)),
        std=float(np.std(array)),
    )


def format_metric_summary(summary: MetricSummary) -> str:
    return f"{summary.mean:.4f} ± {summary.std:.4f}"


def adjust_cv_splits(y: NDArray[np.int64], requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    if counts.size == 0:
        raise ValueError("Empty label array provided for cross-validation")
    max_splits = int(np.min(counts))
    if max_splits < 2:
        raise ValueError("Not enough samples per class for stratified cross-validation")
    return max(2, min(requested, max_splits))


def cross_validate_model(
    factory: Callable[[], Any],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_splits: int,
    random_state: int,
) -> dict[str, MetricSummary]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics_by_fold = {"accuracy": [], "f1_score": [], "roc_auc": []}
    for train_indices, valid_indices in cv.split(X, y):
        model = factory()
        model.fit(X[train_indices], y[train_indices])
        metrics = compute_classification_metrics(model, X[valid_indices], y[valid_indices])
        for key in metrics_by_fold:
            metrics_by_fold[key].append(metrics[key])
    return {key: summarize_metric(values) for key, values in metrics_by_fold.items()}


def evaluate_holdout_models(
    factories: list[tuple[str, Callable[[], Any]]],
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
) -> list[dict[str, Any]]:
    holdout = []
    for name, factory in factories:
        model = factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_classification_metrics(model, X_test, y_test)
        detailed = ModelPerformanceAnalyzer.compute_detailed_metrics(y_test, y_pred, y_proba)
        holdout.append(
            {
                "name": name,
                "model": model,
                "metrics": metrics,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "detailed": detailed,
            }
        )
    return holdout


def print_dataset_insights(
    header: str,
    dataset_stats: dict[str, Any],
    separability: float,
    tablefmt: str,
) -> None:
    distribution = ", ".join(
        f"{int(label)}: {count}" for label, count in sorted(dataset_stats["class_distribution"].items())
    )
    table = [
        ["Samples", dataset_stats["n_samples"]],
        ["Features", dataset_stats["n_features"]],
        ["Classes", dataset_stats["n_classes"]],
        ["Class imbalance", f"{dataset_stats['class_imbalance_ratio']:.2f}"],
        ["High correlations", dataset_stats["high_correlations"]],
        ["Class distribution", distribution],
    ]
    print(f"\n📊 Dataset: {header}")
    print(tabulate(table, headers=["Statistic", "Value"], tablefmt=tablefmt))
    print(f"   • Class separability score: {separability:.4f}\n")


def print_cv_summary(cv_results: dict[str, dict[str, MetricSummary]], tablefmt: str) -> None:
    table = []
    for model_name, summary in cv_results.items():
        table.append(
            [
                model_name,
                format_metric_summary(summary["accuracy"]),
                format_metric_summary(summary["f1_score"]),
                format_metric_summary(summary["roc_auc"]),
            ]
        )
    print("📈 Cross-validation summary")
    print(tabulate(table, headers=["Model", "Accuracy", "F1", "AUROC"], tablefmt=tablefmt))


def print_cv_significance(cv_results: dict[str, dict[str, MetricSummary]]) -> None:
    comparisons = [
        ("Custom Decision Tree", "sklearn Decision Tree"),
        ("Custom Random Forest", "sklearn Random Forest"),
        ("Custom Extra Trees", "sklearn Extra Trees"),
    ]
    print("\n🧪 Statistical comparisons (accuracy)")
    for custom, sklearn in comparisons:
        custom_summary = cv_results[custom]["accuracy"]
        sklearn_summary = cv_results[sklearn]["accuracy"]
        stats = ModelPerformanceAnalyzer.statistical_significance_test(
            np.asarray(custom_summary.values, dtype=float),
            np.asarray(sklearn_summary.values, dtype=float),
            test="paired_ttest",
        )
        mean_diff = custom_summary.mean - sklearn_summary.mean
        print(
            f"   • {custom} vs {sklearn}: mean diff {mean_diff:+.4f} (p={stats['p_value']:.4f})"
        )


def print_holdout_summary(results: list[dict[str, Any]], tablefmt: str) -> None:
    table = []
    for entry in results:
        metrics = entry["metrics"]
        detail = entry["detailed"]
        table.append(
            [
                entry["name"],
                f"{metrics['accuracy']:.4f}",
                f"{metrics['f1_score']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                f"{detail['balanced_accuracy']:.4f}",
                f"{detail['matthews_corrcoef']:.4f}",
            ]
        )
    print("\n📌 Holdout test results")
    print(
        tabulate(
            table,
            headers=["Model", "Accuracy", "F1", "AUROC", "Balanced Acc", "MCC"],
            tablefmt=tablefmt,
        )
    )


def print_best_model_details(
    entry: dict[str, Any],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
    tablefmt: str,
) -> None:
    name = entry["name"]
    detail = entry["detailed"]
    print(f"\n🔍 Detailed holdout analysis for {name}")
    print(f"   • Macro precision: {detail['macro_precision']:.4f}")
    print(f"   • Macro recall: {detail['macro_recall']:.4f}")
    print(f"   • Macro F1: {detail['macro_f1']:.4f}")
    print(f"   • Weighted F1: {detail['weighted_f1']:.4f}")
    print("\n   Confusion matrix:")
    cm_table = [list(row) for row in detail["confusion_matrix"]]
    print(tabulate(cm_table, tablefmt=tablefmt))

    per_class = []
    for idx in range(len(detail["precision_per_class"])):
        per_class.append(
            [
                idx,
                f"{detail['precision_per_class'][idx]:.4f}",
                f"{detail['recall_per_class'][idx]:.4f}",
                f"{detail['f1_per_class'][idx]:.4f}",
                int(detail["support_per_class"][idx]),
            ]
        )
    print("\n   Per-class performance")
    print(
        tabulate(
            per_class,
            headers=["Class", "Precision", "Recall", "F1", "Support"],
            tablefmt=tablefmt,
        )
    )

    trees = getattr(entry["model"], "trees", None)
    if trees:
        predictions_list = [tree.predict(X_test) for tree in trees]
        bias_variance = BiasVarianceAnalyzer.decompose_bias_variance(y_test, predictions_list)
        disagreement = BiasVarianceAnalyzer.ensemble_disagreement(predictions_list)
        print("\n   Ensemble insights")
        print(
            f"   • Bias: {bias_variance['bias']:.4f}, Variance: {bias_variance['variance']:.4f},"
            f" Total error: {bias_variance['total_error']:.4f}"
        )
        print(
            f"   • Disagreement mean: {disagreement['mean_disagreement']:.4f},"
            f" max: {disagreement['max_disagreement']:.4f}, diversity std: {disagreement['diversity_std']:.4f}"
        )


def build_model_factories(resolved_max_features: int, args: argparse.Namespace) -> list[tuple[str, Callable[[], Any]]]:
    factories: list[tuple[str, Callable[[], Any]]] = []
    factories.append(
        (
            "Custom Decision Tree",
            lambda: CustomDecisionTreeClassifier(
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                random_state=args.random_seed,
            ),
        )
    )
    factories.append(
        (
            "Custom Random Forest",
            lambda: CustomRandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                max_features=resolved_max_features,
                random_state=args.random_seed,
            ),
        )
    )
    factories.append(
        (
            "Custom Extra Trees",
            lambda: CustomExtraTreesClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                max_features=resolved_max_features,
                n_thresholds=args.extra_tree_thresholds,
                random_state=args.random_seed,
            ),
        )
    )
    factories.append(
        (
            "sklearn Decision Tree",
            lambda: SklearnDecisionTreeClassifier(
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                random_state=args.random_seed,
            ),
        )
    )
    factories.append(
        (
            "sklearn Random Forest",
            lambda: SklearnRandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                max_features=resolved_max_features,
                random_state=args.random_seed,
            ),
        )
    )
    factories.append(
        (
            "sklearn Extra Trees",
            lambda: SklearnExtraTreesClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                max_features=resolved_max_features,
                random_state=args.random_seed,
            ),
        )
    )
    return factories


def resolve_max_features(option: str, n_features: int) -> int:
    normalized = option.strip().lower()
    if normalized == "sqrt":
        value = math.sqrt(n_features)
    elif normalized == "log2":
        value = math.log2(n_features)
    elif normalized.endswith("%"):
        try:
            percentage = float(normalized[:-1]) / 100
        except ValueError as exc:
            raise ValueError("Invalid percentage for max_features") from exc
        value = n_features * percentage
    else:
        value = float(normalized)
    resolved = max(1, min(n_features, int(round(value))))
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate custom and scikit-learn tree-based learners on Iris and Wine datasets."
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction of data reserved for testing")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum depth of custom trees")
    parser.add_argument("--min-samples-split", type=int, default=4, help="Minimum samples required to split a node")
    parser.add_argument("--n-estimators", type=int, default=60, help="Number of trees in each ensemble")
    parser.add_argument(
        "--max-features",
        type=str,
        default="sqrt",
        help="Max features per split (sqrt/log2/% or positive int)",
    )
    parser.add_argument("--extra-tree-thresholds", type=int, default=5, help="Random thresholds per feature in Extra Trees")
    parser.add_argument("--random-seed", type=int, default=0, help="Seed all randomness for reproducibility")
    parser.add_argument(
        "--table-format",
        type=str,
        default="github",
        choices=["plain", "simple", "github", "grid", "pretty"],
        help="Output table format that suits the terminal",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified cross-validation folds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_specs: list[tuple[str, Literal["iris", "wine"]]] = [("Iris", "iris"), ("Wine", "wine")]

    print("\n" + "=" * 100)
    print("TREE-BASED LEARNERS: CUSTOM IMPLEMENTATIONS vs SCIKIT-LEARN".center(100))
    print("=" * 100 + "\n")

    analyzer = DatasetAnalyzer()

    for header, dataset_key in dataset_specs:
        X_train, X_test, y_train, y_test = load_and_split(
            dataset_key, test_size=args.test_size, random_state=args.random_seed
        )
        X_full = np.vstack((X_train, X_test))
        y_full = np.concatenate((y_train, y_test))
        dataset_stats = analyzer.compute_dataset_stats(X_full, y_full)
        separability = analyzer.compute_separability_score(X_full, y_full)

        print_dataset_insights(header, dataset_stats, separability, args.table_format)

        resolved_max_features = resolve_max_features(args.max_features, X_train.shape[1])
        factories = build_model_factories(resolved_max_features, args)

        cv_splits = adjust_cv_splits(y_train, args.cv_folds)
        cv_results = {
            name: cross_validate_model(factory, X_train, y_train, cv_splits, args.random_seed)
            for name, factory in factories
        }

        print_cv_summary(cv_results, args.table_format)
        print_cv_significance(cv_results)

        holdout_results = evaluate_holdout_models(factories, X_train, y_train, X_test, y_test)
        print_holdout_summary(holdout_results, args.table_format)

        best_entry = max(holdout_results, key=lambda entry: entry["metrics"]["accuracy"])
        print_best_model_details(best_entry, X_test, y_test, args.table_format)

    print("=" * 100)
    print("✅ Experiment completed successfully!".center(100))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
