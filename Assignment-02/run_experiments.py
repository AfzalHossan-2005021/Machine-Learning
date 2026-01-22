from __future__ import annotations

import argparse
import math

from tabulate import tabulate
from typing import Any, Callable

from src.dataset import load_and_split
from src.evaluate import compute_classification_metrics

from src.tree import DecisionTreeClassifier as CustomDecisionTreeClassifier
from src.ensemble import ExtraTreesClassifier as CustomExtraTreesClassifier
from src.ensemble import RandomForestClassifier as CustomRandomForestClassifier

from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier as SklearnExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier



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


def build_model_factories(resolved_max_features: int, args: argparse.Namespace) -> list[tuple[str, Callable[[], Any]]]:
    factories: list[tuple[str, Callable[[], Any]]] = []
    factories.append(
        (
            "Custom Decision Tree",
            lambda: CustomDecisionTreeClassifier(
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                max_features=resolved_max_features,
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
                max_features=resolved_max_features,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate custom and scikit-learn tree-based learners on Iris and Wine datasets."
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction of data reserved for testing")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum depth of custom trees")
    parser.add_argument("--min-samples-split", type=int, default=4, help="Minimum samples required to split a node")
    parser.add_argument("--n-estimators", type=int, default=60, help="Number of trees in each ensemble")
    parser.add_argument("--max-features", type=str, default="sqrt", help="Max features per split (sqrt/log2/% or positive int)")
    parser.add_argument("--extra-tree-thresholds", type=int, default=10, help="Number of random thresholds per feature in Extra Trees")
    parser.add_argument("--random-seed", type=int, default=0, help="Seed all randomness for reproducibility")
    parser.add_argument(
        "--table-format",
        type=str,
        default="github",
        choices=["plain", "simple", "github", "grid", "pretty"],
        help="Output table format that suits the terminal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_specs = [("Iris", "iris"), ("Wine", "wine")]

    for header, dataset_key in dataset_specs:
        X_train, X_test, y_train, y_test = load_and_split(
            dataset_key, test_size=args.test_size, random_state=args.random_seed
        )
        resolved_max_features = resolve_max_features(args.max_features, X_train.shape[1])
        factories = build_model_factories(resolved_max_features, args)
        results: list[tuple[str, float, float, float]] = []

        for model_name, factory in factories:
            model = factory()
            model.fit(X_train, y_train)
            metrics = compute_classification_metrics(model, X_test, y_test)
            results.append(
                (
                    model_name,
                    metrics["accuracy"],
                    metrics["f1_score"],
                    metrics["roc_auc"],
                )
            )

        print(f"Dataset: {header} ({X_train.shape[0] + X_test.shape[0]} samples, {X_train.shape[1]} features)")
        print(tabulate(results, headers=["Model", "Accuracy", "F1", "AUROC"], tablefmt=args.table_format, floatfmt=".4f"))
        print("\n")


if __name__ == "__main__":
    main()
