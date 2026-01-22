from __future__ import annotations

import numpy as np

from typing import Protocol, cast
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # type: ignore


class ProbabilisticClassifier(Protocol):
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:  # pragma: no cover - typing only
        ...

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:  # pragma: no cover - typing only
        ...


def compute_classification_metrics(
    model: ProbabilisticClassifier, X: NDArray[np.float64], y: NDArray[np.int64]
) -> dict[str, float]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    if y_proba.ndim == 1 or y_proba.shape[1] == 1:
        roc_auc = float(roc_auc_score(y, y_proba.ravel()))
    else:
        roc_auc = cast(float, roc_auc_score(y, y_proba, multi_class="ovo", average="macro"))
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_score": cast(float, f1_score(y, y_pred, average="macro")),
        "roc_auc": float(roc_auc),
    }
