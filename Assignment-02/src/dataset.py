"""Module for loading and splitting datasets."""

from __future__ import annotations

import numpy as np

from sklearn import datasets
from numpy.typing import NDArray
from typing import Callable, Mapping, Final, Tuple, Literal, Protocol, cast

DatasetSplit = Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]
DatasetName = Literal["iris", "wine"]


class _SklearnBundle(Protocol):
    data: NDArray[np.float64]
    target: NDArray[np.int64]


_DATASET_LOADERS: Final[Mapping[DatasetName, Callable[..., _SklearnBundle]]] = {
    "iris": cast(Callable[..., _SklearnBundle], datasets.load_iris),
    "wine": cast(Callable[..., _SklearnBundle], datasets.load_wine),
}


# implement our own train_test_split to avoid sklearn dependency
def train_test_split(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    test_size: float = 0.25,
    stratify: NDArray[np.int64] | None = None,
    random_state: int | None = None,
) -> DatasetSplit:
    """Split arrays or matrices into random train and test subsets."""
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0.0 and 1.0")

    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    if stratify is not None:
        unique_classes, y_indices = np.unique(stratify, return_inverse=True)
        train_indices: list[int] = []
        test_indices: list[int] = []
        for cls in unique_classes:
            cls_mask = (y_indices == cls)
            cls_indices = np.where(cls_mask)[0]
            rng.shuffle(cls_indices)
            n_cls_samples = cls_indices.size
            n_cls_test = int(n_cls_samples * test_size)
            test_indices.extend(cls_indices[:n_cls_test].tolist())
            train_indices.extend(cls_indices[n_cls_test:].tolist())
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
    else:
        all_indices = np.arange(n_samples)
        rng.shuffle(all_indices)
        test_indices = all_indices[:n_test].tolist()
        train_indices = all_indices[n_test:].tolist()

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def load_and_split(
    name: DatasetName,
    test_size: float = 0.25,
    random_state: int | None = None,
) -> DatasetSplit:
    """Load the requested dataset and return a train/test split."""
    loader = _DATASET_LOADERS[name]
    bundle: _SklearnBundle = loader()

    x = np.asarray(bundle.data, dtype=np.float64)
    y = np.asarray(bundle.target, dtype=np.int64)

    return train_test_split(x, y, test_size, y, random_state)
