from __future__ import annotations

from typing import Protocol, cast, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # type: ignore


class ProbabilisticClassifier(Protocol):
    """Protocol for classifiers that provide discrete predictions and probabilities.

    Implementations must provide both `predict` and `predict_proba` methods.
    """
    def predict(
        self, X: NDArray[np.float64]
    ) -> NDArray[np.int64]:  # pragma: no cover - typing only
        ...

    def predict_proba(
        self, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:  # pragma: no cover - typing only
        ...


def compute_classification_metrics(
    model: ProbabilisticClassifier,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    f1_average: Literal["binary", "micro", "macro", "weighted", "samples"] = "macro",
    roc_method: Literal["ovo", "ovr"] = "ovo",
) -> dict[str, float]:
    """Compute common classification metrics.

    Parameters
    - model: any object implementing `predict` and `predict_proba`.
    - X, y: numpy arrays of features and integer labels.
    - f1_average: passed to `sklearn.metrics.f1_score` (e.g. 'micro','macro','weighted','binary').
    - roc_method: one of 'ovo' or 'ovr' and forwarded to `roc_auc_score` for multiclass.

    Returns
    A dict with keys: ``accuracy``, ``f1_score``, ``roc_auc``.

    Raises
    - ValueError on shape mismatches, single-class `y`, or invalid parameters.
    """
    # basic validations
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    allowed_f1 = {"binary", "micro", "macro", "weighted", "samples"}
    if f1_average not in allowed_f1:
        raise ValueError(
            f"f1_average must be one of {sorted(allowed_f1)}, got {f1_average!r}"
        )

    if roc_method not in ("ovo", "ovr"):
        raise ValueError("roc_method must be either 'ovo' or 'ovr'")

    # ensure there is more than one class in y
    unique_labels = np.unique(y)
    if unique_labels.size <= 1:
        raise ValueError("y contains a single class — metrics like ROC AUC are undefined")

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # ROC AUC: handle binary (1d or single-column) vs multiclass
    if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 1):
        roc_auc = float(roc_auc_score(y, y_proba.ravel()))
    else:
        roc_auc = cast(
            float, roc_auc_score(y, y_proba, multi_class=roc_method, average="macro")
        )

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_score": cast(float, f1_score(y, y_pred, average=f1_average)),
        "roc_auc": float(roc_auc),
    }
