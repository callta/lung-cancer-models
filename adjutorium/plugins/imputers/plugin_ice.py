# stdlib
import time
from typing import Any, List, Union

# third party
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.imputers.base as base
import adjutorium.utils.serialization as serialization


class IterativeChainedEquationsPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Multivariate Iterative chained equations Imputation strategy.

    Method:
        Multivariate Iterative chained equations(MICE) methods model each feature with missing values as a function of other features in a round-robin fashion. For each step of the round-robin imputation, we use a BayesianRidge estimator, which does a regularized linear regression.

    Args:
        max_iter: int, default=500
            maximum number of imputation rounds to perform.
        random_state: int, default set to the current time.
            seed of the pseudo random number generator to use.

    Example:
        >>> import numpy as np
        >>> from adjutorium.plugins.imputers import Imputers
        >>> plugin = Imputers().get("ice")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
                  0         1         2         3
        0  1.000000  1.000000  1.000000  1.000000
        1  1.333333  1.666667  1.666667  1.333333
        2  1.000000  2.000000  2.000000  1.000000
        3  2.000000  2.000000  2.000000  2.000000
    """

    def __init__(
        self,
        max_iter: int = 10000,
        random_state: Union[int, None] = None,
        model: Any = None,
    ) -> None:
        super().__init__()

        if model:
            self._model = model
            return

        if not random_state:
            random_state = int(time.time())

        self._model = IterativeImputer(
            random_state=random_state,
            max_iter=max_iter,
            sample_posterior=False,
        )

    @staticmethod
    def name() -> str:
        return "ice"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "IterativeChainedEquationsPlugin":
        self._model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self._model)

    @classmethod
    def load(cls, buff: bytes) -> "IterativeChainedEquationsPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = IterativeChainedEquationsPlugin
