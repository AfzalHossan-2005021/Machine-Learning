"""Module for loading and splitting datasets."""

from __future__ import annotations

from typing import Callable, Dict, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn import datasets  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.utils import Bunch  # type: ignore[import-untyped]

DatasetSplit = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]
]
DatasetName = Literal["iris", "wine"]


_DATASET_LOADERS: Dict[DatasetName, Callable[..., Bunch]] = {
    "iris": datasets.load_iris,
    "wine": datasets.load_wine,
}


def load_and_split(
    name: DatasetName,
    test_size: float = 0.25,
    random_state: int | None = None,
) -> DatasetSplit:
    """Load the requested dataset and return a train/test split."""
    loader: Callable[..., Bunch] = _DATASET_LOADERS[name]

    x = np.asarray(loader().data, dtype=np.float64)
    y = np.asarray(loader().target, dtype=np.int64)

    return train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
