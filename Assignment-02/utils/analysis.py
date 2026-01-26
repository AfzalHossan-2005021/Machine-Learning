"""Advanced statistical analysis and visualization utilities."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray
from scipy import stats  # type: ignore
from sklearn.metrics import (  # type: ignore
    confusion_matrix,
    precision_recall_fscore_support
)
import warnings

warnings.filterwarnings("ignore")


class ModelPerformanceAnalyzer:
    """Comprehensive analysis of model performance metrics."""

    @staticmethod
    def compute_detailed_metrics(
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.int64],
        y_proba: NDArray[np.float64],
    ) -> Dict[str, Any]:
        """Compute detailed classification metrics including per-class analysis."""
        n_classes = len(np.unique(y_true))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=np.arange(n_classes)
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
        
        # Balanced accuracy
        recalls = cm.diagonal() / cm.sum(axis=1)
        balanced_acc = np.mean(recalls)
        
        # Matthews correlation coefficient
        mcc = ModelPerformanceAnalyzer._matthews_corrcoef(y_true, y_pred)
        
        return {
            "confusion_matrix": cm,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": f1,
            "support_per_class": support,
            "balanced_accuracy": float(balanced_acc),
            "matthews_corrcoef": mcc,
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "weighted_f1": float(np.average(f1, weights=support)),
        }

    @staticmethod
    def _matthews_corrcoef(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
        """Compute Matthews Correlation Coefficient."""
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if denominator == 0:
                return 0.0
            return float((tp * tn - fp * fn) / denominator)
        
        # For multiclass, use macro average approach
        mccs = []
        for i in range(n_classes):
            y_binary_true = (y_true == i).astype(int)
            y_binary_pred = (y_pred == i).astype(int)
            tp = np.sum((y_binary_true == 1) & (y_binary_pred == 1))
            tn = np.sum((y_binary_true == 0) & (y_binary_pred == 0))
            fp = np.sum((y_binary_true == 0) & (y_binary_pred == 1))
            fn = np.sum((y_binary_true == 1) & (y_binary_pred == 0))
            
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if denominator > 0:
                mccs.append((tp * tn - fp * fn) / denominator)
        
        return float(np.mean(mccs)) if mccs else 0.0

    @staticmethod
    def statistical_significance_test(
        scores_model1: NDArray[np.float64],
        scores_model2: NDArray[np.float64],
        test: str = "paired_ttest",
    ) -> Dict[str, float]:
        """Perform statistical significance tests between two models."""
        if len(scores_model1) != len(scores_model2):
            raise ValueError("Score arrays must have the same length")
        
        if test == "paired_ttest":
            t_stat, p_value = stats.ttest_rel(scores_model1, scores_model2)
        elif test == "wilcoxon":
            t_stat, p_value = stats.wilcoxon(scores_model1, scores_model2)
        elif test == "mannwhitneyu":
            t_stat, p_value = stats.mannwhitneyu(scores_model1, scores_model2)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        return {
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "is_significant": p_value < 0.05,
            "mean_diff": float(np.mean(scores_model1 - scores_model2)),
            "std_diff": float(np.std(scores_model1 - scores_model2)),
        }


class BiasVarianceAnalyzer:
    """Analyze bias-variance tradeoff in ensemble models."""

    @staticmethod
    def decompose_bias_variance(
        y_true: NDArray[np.int64],
        predictions_list: List[NDArray[np.int64]],
    ) -> Dict[str, float]:
        """Decompose MSE into bias and variance components."""
        predictions_array = np.array(predictions_list)  # shape: (n_models, n_samples)
        
        # Main prediction (average/majority vote)
        main_pred = np.round(np.mean(predictions_array, axis=0)).astype(int)
        
        # 0-1 loss (for classification)
        loss_main = np.mean(main_pred != y_true)
        
        # Bias: error of main predictor
        bias = loss_main
        
        # Variance: average error of individual predictors
        individual_losses = np.mean(predictions_array != y_true[None, :], axis=1)
        variance = np.mean(individual_losses) - bias
        variance = max(0, variance)  # Ensure non-negative
        
        return {
            "bias": float(bias),
            "variance": float(variance),
            "total_error": float(bias + variance),
        }

    @staticmethod
    def ensemble_disagreement(
        predictions_list: List[NDArray[np.int64]],
    ) -> Dict[str, float]:
        """Compute disagreement metrics among ensemble members."""
        predictions_array = np.array(predictions_list)
        n_models, n_samples = predictions_array.shape
        
        if n_models < 2:
            return {"mean_disagreement": 0.0, "max_disagreement": 0.0}
        
        # Pairwise disagreement
        disagreements = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = np.mean(predictions_array[i] != predictions_array[j])
                disagreements.append(disagreement)
        
        mean_disagreement = float(np.mean(disagreements))
        max_disagreement = float(np.max(disagreements))
        
        # Diversity measure: Kappa statistic
        diversity = np.std([np.mean(predictions_array[i]) for i in range(n_models)])
        
        return {
            "mean_disagreement": mean_disagreement,
            "max_disagreement": max_disagreement,
            "diversity_std": float(diversity),
        }


class DatasetAnalyzer:
    """Analyze dataset properties."""

    @staticmethod
    def compute_dataset_stats(X: NDArray[np.float64], y: NDArray[np.int64]) -> Dict[str, Any]:
        """Compute comprehensive dataset statistics."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        class_imbalance = np.max(counts) / np.min(counts)
        
        # Feature statistics
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        feature_ranges = np.max(X, axis=0) - np.min(X, axis=0)
        
        # Feature correlations (simplified)
        feature_corr = np.corrcoef(X.T)
        high_corr = np.sum(np.abs(feature_corr) > 0.9) // 2 - n_features
        
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "class_distribution": class_dist,
            "class_imbalance_ratio": float(class_imbalance),
            "feature_means": feature_means.tolist(),
            "feature_stds": feature_stds.tolist(),
            "feature_ranges": feature_ranges.tolist(),
            "high_correlations": int(high_corr),
        }

    @staticmethod
    def compute_separability_score(X: NDArray[np.float64], y: NDArray[np.int64]) -> float:
        """Compute a measure of class separability using within/between-class distances."""
        classes = np.unique(y)
        
        between_class_dist = 0
        within_class_dist = 0
        
        for i, c1 in enumerate(classes):
            X_c1 = X[y == c1]
            center_c1 = np.mean(X_c1, axis=0)
            
            # Within-class distance
            within_class_dist += np.mean(np.linalg.norm(X_c1 - center_c1, axis=1))
            
            # Between-class distance
            for j, c2 in enumerate(classes):
                if j > i:
                    center_c2 = np.mean(X[y == c2], axis=0)
                    between_class_dist += np.linalg.norm(center_c1 - center_c2)
        
        if within_class_dist == 0:
            return 0.0
        return float(between_class_dist / within_class_dist)
