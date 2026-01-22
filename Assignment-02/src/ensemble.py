from __future__ import annotations

import numpy as np

from typing import Sequence
from numpy.typing import NDArray

from .tree import DecisionTreeClassifier, ExtraTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = 60,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | None = None,
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = max(1, n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.trees: list[DecisionTreeClassifier] = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "RandomForestClassifier":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        n_samples = X_arr.shape[0]
        self.trees = []
        for _ in range(self.n_estimators):
            indices: NDArray[np.intp] = (
                self._bootstrap_indices(n_samples) if self.bootstrap else np.arange(n_samples, dtype=np.intp)
            )
            tree_random_state = self._rng.integers(0, np.iinfo(np.int32).max)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=int(tree_random_state),
            )
            tree.fit(X_arr[indices], y_arr[indices])
            self.trees.append(tree)
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> NDArray[np.float64]:
        if not self.trees:
            raise ValueError("Fit the random forest before prediction")
        probs = np.stack([tree.predict_proba(X) for tree in self.trees])
        return np.mean(probs, axis=0)

    def predict(self, X: Sequence[Sequence[float]]) -> NDArray[np.int64]:
        return np.argmax(self.predict_proba(X), axis=1)

    def _bootstrap_indices(self, n_samples: int) -> NDArray[np.intp]:
        return self._rng.integers(0, n_samples, size=n_samples)


class ExtraTreesClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = 60,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | None = None,
        n_thresholds: int = 10,
        bootstrap: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = max(1, n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_thresholds = max(1, n_thresholds)
        self.bootstrap = bootstrap
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.trees: list[ExtraTreeClassifier] = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "ExtraTreesClassifier":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        n_samples = X_arr.shape[0]
        self.trees = []
        for _ in range(self.n_estimators):
            indices: NDArray[np.intp] = (
                self._bootstrap_indices(n_samples) if self.bootstrap else np.arange(n_samples, dtype=np.intp)
            )
            tree_random_state = self._rng.integers(0, np.iinfo(np.int32).max)
            tree = ExtraTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                random_state=int(tree_random_state),
            )
            tree.fit(X_arr[indices], y_arr[indices])
            self.trees.append(tree)
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> NDArray[np.float64]:
        if not self.trees:
            raise ValueError("Fit the ensemble before prediction")
        probs = np.stack([tree.predict_proba(X) for tree in self.trees])
        return np.mean(probs, axis=0)

    def predict(self, X: Sequence[Sequence[float]]) -> NDArray[np.int64]:
        return np.argmax(self.predict_proba(X), axis=1)

    def _bootstrap_indices(self, n_samples: int) -> NDArray[np.intp]:
        return self._rng.integers(0, n_samples, size=n_samples)
