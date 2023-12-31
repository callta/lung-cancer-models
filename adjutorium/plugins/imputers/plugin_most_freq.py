# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.impute import SimpleImputer

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.imputers.base as base
import adjutorium.utils.serialization as serialization


class MostFrequentPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Most Frequent Imputation strategy.

    Method:
        The Most Frequent Imputation strategy replaces the missing using the most frequent value along each column.

    Example:
        >>> import numpy as np
        >>> from adjutorium.plugins.imputers import Imputers
        >>> plugin = Imputers().get("most_frequent")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
             0    1    2    3
        0  1.0  1.0  1.0  1.0
        1  1.0  2.0  2.0  1.0
        2  1.0  2.0  2.0  1.0
        3  2.0  2.0  2.0  2.0
    """

    def __init__(self, model: Any = None) -> None:
        super().__init__()

        if model:
            self._model = model
            return

        self._model = SimpleImputer(strategy="most_frequent")

    @staticmethod
    def name() -> str:
        return "most_frequent"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "MostFrequentPlugin":
        self._model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self._model)

    @classmethod
    def load(cls, buff: bytes) -> "MostFrequentPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = MostFrequentPlugin
