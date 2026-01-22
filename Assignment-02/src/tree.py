from __future__ import annotations

import numpy as np

from typing import Iterable, Any
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class TreeNode:
    feature_index: int | None = None
    threshold: float | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None
    value: int | None = None
    class_counts: NDArray[Any] | None = None
    samples: int = 0


class DecisionTreeClassifier:
    def __init__(
        self,
        *,
        max_depth: int | None = None,       # Maximum number of levels in each decision tree
        min_samples_split: int = 2,         # Minimum number of samples required to split an internal node
        max_features: int | None = None,    # Maximum number of features in randomly selected subset of features to chooses the best split only from that subset
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)
        self.max_features = max_features
        self.n_classes_: int | None = None      # Number of classes observed during fitting
        self.n_features_in_: int | None = None  # Number of features observed during fitting
        self.root: TreeNode | None = None       # Root node of the decision tree
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[int]) -> "DecisionTreeClassifier":
        X_arr, y_arr = self._preprocess_data(X, y)

        self.n_features_in_ = X_arr.shape[1]
        self.n_classes_ = int(np.unique(y_arr).size)
        self.root = self._build_tree(X_arr, y_arr, depth=0)
        return self

    def predict(self, X: Iterable[Iterable[float]]) -> NDArray[Any]:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: Iterable[Iterable[float]]) -> NDArray[Any]:
        if self.root is None or self.n_classes_ is None:
            raise ValueError("The tree has not been fitted yet")
        X_arr = np.asarray(X, dtype=float)
        probs = np.zeros((X_arr.shape[0], self.n_classes_), dtype=float)
        for idx, sample in enumerate(X_arr):
            node = self.root
            while node is not None and node.feature_index is not None:
                if sample[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            if node is None or node.class_counts is None:
                continue
            total = float(node.class_counts.sum())
            if total <= 0:
                probs[idx, node.value or 0] = 1.0
            else:
                probs[idx] = node.class_counts / total
        return probs
    
    def _preprocess_data(self, X: Iterable[Iterable[float]], y: Iterable[int]) -> tuple[NDArray[Any], NDArray[Any]]:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2d array, got shape {X_arr.shape}")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X and y must have the same number of samples, got {X_arr.shape[0]} and {y_arr.shape[0]}")
        return X_arr, y_arr

    def _build_tree(self, X: NDArray[Any], y: NDArray[Any], depth: int) -> TreeNode:
        if (
            X.shape[0] < self.min_samples_split     # minimum samples required to split
            or self.max_depth is not None and depth >= self.max_depth   # maximum depth reached
            or np.unique(y).size == 1   # all samples belong to the same class
        ):
            return self._make_leaf(y)

        split = self._find_best_split(X, y)
        if split is None:
            return self._make_leaf(y)
        
        node = TreeNode(samples=X.shape[0])

        node.feature_index = int(split["feature_index"])
        node.threshold = float(split["threshold"])
        left_indices = split["left_mask"]
        right_indices = split["right_mask"]
        node.left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        node.right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return node

    def _make_leaf(self, y: NDArray[Any]) -> TreeNode:
        if(y.size == 0):
            return TreeNode(value=None, class_counts=None, samples=0)
        assert self.n_classes_ is not None
        counts = np.bincount(y, minlength=self.n_classes_)
        majority_class = int(np.argmax(counts))
        return TreeNode(value=majority_class, class_counts=counts, samples=y.size)

    def _find_best_split(self, X: NDArray[Any], y: NDArray[Any]) -> dict[str, Any] | None:
        best_gain = -np.inf
        best_split = None
        current_impurity = self._entropy(y)
        n_features = X.shape[1]
        feature_indices = self._feature_indices(n_features)

        for feature in feature_indices:
            thresholds = self._generate_thresholds(X[:, feature], y)
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                n_left = int(left_mask.sum())
                n_right = int(right_mask.sum())
                if n_left == 0 or n_right == 0:
                    continue
                y_left = y[left_mask]
                y_right = y[right_mask]
                impurity = (n_left * self._entropy(y_left) + n_right * self._entropy(y_right)) / (n_left + n_right)
                gain = current_impurity - impurity
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature_index": feature,
                        "threshold": threshold,
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                    }
        return best_split

    def _feature_indices(self, n_features: int) -> NDArray[Any]:
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)
        return self._rng.choice(n_features, size=self.max_features, replace=False)
    
    def _generate_thresholds(self, feature_values: NDArray[np.floating], y: NDArray[np.integer]) -> NDArray[np.floating]:
        # 1. When there are less than 2 samples, no thresholds can be generated
        if feature_values.size < 2:
            return np.array([], dtype=float)

        # 2. Identify valid indices where feature values are not NaN
        valid_mask = ~np.isnan(feature_values)

        # Case 1: all values missing
        if not np.any(valid_mask):
            return np.array([], dtype=float)

        # Case 2: split only on valid values
        x_valid = feature_values[valid_mask]
        y_valid = y[valid_mask]

        return self._generate_boundary_thresholds(x_valid, y_valid)

    def _generate_boundary_thresholds(self, feature_values: NDArray[np.floating], y: NDArray[np.integer]) -> NDArray[np.floating]:
        # 1. Sort X and y (O(N log N))
        sort_indices = np.argsort(feature_values)
        x_sorted = feature_values[sort_indices]
        y_sorted = y[sort_indices]

        # 2. Identify transitions where the label changes AND the feature value progresses
        # boundary_mask: y changes between i and i+1
        # value_mask: x actually increases (prevents splitting identical x values)
        boundary_mask = y_sorted[:-1] != y_sorted[1:]
        value_mask = x_sorted[:-1] < x_sorted[1:]

        valid = boundary_mask & value_mask

        if not np.any(valid):
            return np.array([], dtype=float)

        # 3. Calculate midpoints using overflow-safe formula
        left = x_sorted[:-1][valid]
        right = x_sorted[1:][valid]

        thresholds = left + (right - left) / 2.0

        return np.unique(thresholds)

    
    @staticmethod
    def _entropy(y: NDArray[Any]) -> float:
        if y.size == 0:
            return 0.0
        counts = np.bincount(y) # counts the occurrences of each non-negative integer value in the input 1-dimensional array
        probabilities = counts / y.size
        nonzero_probs = probabilities[probabilities > 0]
        return -np.sum(nonzero_probs * np.log2(nonzero_probs))


class ExtraTreeClassifier(DecisionTreeClassifier):
    def __init__(
        self,
        *,
        n_thresholds: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_thresholds = max(1, n_thresholds)

    def _generate_thresholds(
        self,
        feature_values: NDArray[np.floating],
        y: NDArray[np.integer]
    ) -> NDArray[np.floating]:

        f_min = np.nanmin(feature_values)
        f_max = np.nanmax(feature_values)

        if np.isnan(f_min) or f_min >= f_max:
            return np.array([], dtype=float)

        eps = np.finfo(feature_values.dtype).eps
        low = f_min + eps
        high = f_max - eps

        if low >= high:
            return np.array([], dtype=float)

        return self._rng.uniform(low, high, size=self.n_thresholds)

