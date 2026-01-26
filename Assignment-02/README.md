# Tree-Based Learners From Scratch: Comprehensive Analysis

## Overview

This project implements decision trees, random forests, and extremely randomized trees (Extra Trees) without relying on scikit-learn's tree implementations. The implementation includes comprehensive statistical analysis, cross-validation studies, and detailed performance comparisons against scikit-learn's optimized implementations.

## Key Features

✨ **Custom Implementations:**

- Decision Tree Classifier with entropy-based splits
- Random Forest Classifier with bootstrap aggregation
- Extra Trees Classifier with random threshold selection
- Configurable hyperparameters (depth, min_samples_split, max_features)
- Full support for probability predictions

🎯 **Comprehensive Analysis:**

- 5-fold cross-validation for robustness assessment
- Statistical significance testing (paired t-test, Wilcoxon)
- Hyperparameter sensitivity analysis
- Dataset properties analysis
- Class separability scoring

📊 **Evaluation Metrics:**

- Accuracy, F1-score, AUROC
- Per-class precision, recall, and F1
- Balanced accuracy
- Matthews correlation coefficient

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic experiments
python run_experiments.py

# Run with custom parameters
python run_experiments.py --max-depth 12 --n-estimators 80 --random-seed 42

# View detailed analysis
jupyter notebook analysis_notebook.ipynb
```

## Project Structure

```
├── custom/
│   ├── tree.py              # Decision Tree & Extra Tree implementations
│   └── ensemble.py          # Random Forest & Extra Trees ensemble
├── utils/
│   ├── dataset.py           # Dataset loading and splitting
│   ├── evaluate.py          # Metrics computation
│   └── analysis.py          # Statistical analysis tools
├── run_experiments.py       # Main CLI entry point
├── analysis_notebook.ipynb  # Comprehensive Jupyter notebook analysis
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Algorithm Details

### Decision Tree Classifier

- **Split Criterion:** Information Gain (Entropy-based)
- **Feature Selection:** Optional random feature subsampling
- **Stopping Criteria:** Maximum depth, minimum samples per split, class purity
- **Leaf Values:** Majority class with probability estimates

### Random Forest Classifier

- **Ensemble Method:** Bootstrap aggregation (Bagging)
- **Base Learners:** Custom Decision Trees
- **Aggregation:** Majority voting with probability averaging
- **Randomness:** Bootstrap samples + feature subsampling

### Extra Trees (Extremely Randomized Trees)

- **Key Difference:** Random threshold selection instead of optimal search
- **Efficiency:** Faster training due to reduced threshold computation
- **Variance:** Higher variance but potentially better bias-variance tradeoff
- **Diversity:** Enhanced diversity through random thresholds

## Configuration Options

```bash
python run_experiments.py [OPTIONS]

Options:
  --test-size FLOAT              Fraction of data for testing (default: 0.25)
  --max-depth INT                Maximum tree depth (default: 10)
  --min-samples-split INT        Minimum samples to split (default: 4)
  --n-estimators INT             Trees in ensemble (default: 60)
  --max-features STR             Features per split: sqrt/log2/% or int (default: sqrt)
  --extra-tree-thresholds INT    Random thresholds in Extra Trees (default: 5)
  --random-seed INT              Reproducibility seed (default: 0)
  --table-format STR             Output format: plain/simple/github/grid/pretty (default: github)
```

## Performance Results

### Iris Dataset

Excellent performance on this well-separated dataset:

- Custom & Sklearn models achieve >90% accuracy
- Very high generalization on both metrics
- Strong F1 and AUROC scores

### Wine Dataset

Good performance on moderately separated dataset:

- Ensemble methods significantly outperform single trees
- More challenging due to feature correlations
- Demonstrates importance of model selection

## Advanced Analysis Features

### Statistical Testing

- Paired t-tests comparing custom vs sklearn implementations
- Wilcoxon signed-rank tests (non-parametric alternative)
- P-value computation for statistical significance

### Cross-Validation

- Stratified K-Fold cross-validation (default: 5 folds)
- Robustness assessment across different train/test splits
- Model consistency evaluation

### Hyperparameter Sensitivity

- Max depth impact on accuracy
- Ensemble size effects
- Feature subsampling influence

### Dataset Analysis

- Class distribution analysis
- Feature correlation matrices
- Class separability scoring
- Skewness and kurtosis computation

## Implementation Highlights

### Type Safety

- Full type hints for all functions and methods
- Protocol-based interfaces for polymorphism
- NumPy type annotations

### Code Quality

- Comprehensive docstrings
- Clean separation of concerns
- Reusable utility functions
- No external tree implementations used

### Reproducibility

- Seeded random number generation
- Deterministic algorithms
- Full hyperparameter control

## Key Findings

✅ **Custom implementations are competitive** with scikit-learn
✅ **No statistically significant differences** detected in performance
✅ **Ensemble methods greatly improve** single tree performance
✅ **Hyperparameter tuning** is crucial for optimal results
✅ **Good generalization** on both datasets with proper cross-validation

## Usage Examples

```python
from custom.tree import DecisionTreeClassifier
from custom.ensemble import RandomForestClassifier, ExtraTreesClassifier
from utils.dataset import load_and_split
from utils.evaluate import compute_classification_metrics

# Load data
X_train, X_test, y_train, y_test = load_and_split('iris', test_size=0.25)

# Train single tree
tree = DecisionTreeClassifier(max_depth=10, min_samples_split=4)
tree.fit(X_train, y_train)
metrics = compute_classification_metrics(tree, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Train random forest
forest = RandomForestClassifier(n_estimators=80, max_depth=12)
forest.fit(X_train, y_train)
metrics = compute_classification_metrics(forest, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Train extra trees
extra = ExtraTreesClassifier(n_estimators=80, n_thresholds=10)
extra.fit(X_train, y_train)
metrics = compute_classification_metrics(extra, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Advanced Analysis Notebook

The `analysis_notebook.ipynb` provides:

1. **Data Exploration** - Dataset properties and distributions
2. **Descriptive Statistics** - Mean, std, skewness, kurtosis
3. **EDA Visualizations** - Distributions, class balance, feature relationships
4. **Correlation Analysis** - Feature relationships and multicollinearity
5. **Model Comparison** - Side-by-side performance evaluation
6. **Cross-Validation** - Robustness and generalization assessment
7. **Statistical Tests** - Hypothesis validation and significance testing
8. **Hyperparameter Analysis** - Sensitivity to different configurations
9. **Comprehensive Summary** - Key insights and recommendations

## Scientific Rigor

This project demonstrates:

- ✓ Proper train/test splitting with stratification
- ✓ Appropriate metrics for imbalanced datasets
- ✓ Statistical testing with significance levels
- ✓ Cross-validation for robust estimates
- ✓ Reproducibility through seeded randomness
- ✓ Comprehensive documentation

## Requirements

- Python 3.8+
- numpy >= 1.26.0
- scikit-learn >= 1.4.0
- tabulate >= 0.9.0
- scipy >= 1.10.0 (for advanced analysis)
- matplotlib >= 3.5.0 (for notebooks)
- seaborn >= 0.12.0 (for notebooks)

## Notes

- All randomness is controlled via `--random-seed` for reproducibility
- Custom implementations do not use scikit-learn's tree classes
- Proper handling of edge cases (empty splits, missing values)
- Floating-point accuracy considerations for threshold comparisons
- Efficient entropy computation using NumPy operations

## Performance Characteristics

| Aspect           | Custom       | Sklearn      |
| ---------------- | ------------ | ------------ |
| Accuracy         | ✓ Comparable | ✓ Comparable |
| Speed            | ✓ Good       | ✓ Optimized  |
| Memory           | ✓ Efficient  | ✓ Efficient  |
| Features         | ✓ Complete   | ✓ Complete   |
| Interpretability | ✓ High       | ✓ High       |

## Contributing

This implementation serves as an educational resource. Areas for enhancement:

- Tree visualization and interpretation
- Additional split criteria (Gini, log loss)
- Pruning strategies
- Feature importance ranking
- Parallel processing for large ensembles

## License

Educational use

## References

- Breiman, L., et al. "Classification and Regression Trees" (1984)
- Breiman, L. "Random Forests" (2001)
- Geurts, P., Ernst, D., Wehenkel, L. "Extremely randomized trees" (2006)
