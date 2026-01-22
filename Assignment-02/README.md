# Tree-Based Learners From Scratch

This project implements decision trees, random forests, and extremely randomized trees (Extra Trees) without relying on scikit-learn's tree implementations. The intention is to compare the custom learners against scikit-learn's `DecisionTreeClassifier`, `RandomForestClassifier`, and `ExtraTreesClassifier` on the Iris and Wine datasets.

## Features

- Custom `DecisionTreeClassifier` built with Gini impurity, configurable depth/min samples, and feature subsetting.
- `RandomForestClassifier` and `ExtraTreesClassifier` ensembles that reuse the custom tree builders for consistency.
- Evaluation harness that reports accuracy, macro F1, and macro AUROC for both Iris and Wine datasets.
- CLI flags to tune hyperparameters and output format.

## Getting started

```bash
pip install -r requirements.txt
python run_experiments.py --max-depth 12 --n-estimators 80
```

The script prints labeled tables per dataset so you can directly compare the custom learners with their scikit-learn counterparts.

## Project layout

- `src/tree.py`: Core decision tree logic and ExtraTree subclass with random thresholds.
- `src/ensemble.py`: Custom random forest and extra-trees ensembles.
- `src/dataset.py`: Helpers for loading and splitting Iris/Wine.
- `src/evaluate.py`: Metric computation for accuracy, macro F1, and AUROC.
- `run_experiments.py`: CLI entry point that trains every model on each dataset and reports results.
- `requirements.txt`: Minimal dependencies.

## Notes

- All randomness is controlled via the `--random-seed` flag for reproducibility.
- Use `--table-format` to adjust how tables render in your terminal (choices include `plain`, `simple`, `github`, `grid`, and `pretty`).
- The custom learners follow the specification exactly: they do not call into scikit-learn's tree classes and support configurable hyperparameters for depth, splits, and feature sampling.
