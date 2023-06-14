# stdlib
import time
from typing import Any, List, Union

# third party
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.imputers.base as base
import adjutorium.plugins.utils.decorators as decorators
import adjutorium.utils.serialization as serialization


class MissForestPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the MissForest strategy.

    Method:
        Iterative chained equations(ICE) methods model each feature with missing values as a function of other features in a round-robin fashion. For each step of the round-robin imputation, we use a ExtraTreesRegressor, which fits a number of randomized extra-trees and averages the results.

    Args:
        n_estimators: int, default=10
            The number of trees in the forest.
        max_iter: int, default=500
            maximum number of imputation rounds to perform.
        random_state: int, default set to the current time.
            seed of the pseudo random number generator to use.

    Adjutorium Hyperparameters:
        n_estimators: The number of trees in the forest.

    Example:
        >>> import numpy as np
        >>> from adjutorium.plugins.imputers import Imputers
        >>> plugin = Imputers().get("missforest")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
             0    1    2    3
        0  1.0  1.0  1.0  1.0
        1  1.0  1.9  1.9  1.0
        2  1.0  2.0  2.0  1.0
        3  2.0  2.0  2.0  2.0
    """

    def __init__(
        self,
        n_estimators: int = 10,
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

        estimator_rf = ExtraTreesRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        self._model = IterativeImputer(
            estimator=estimator_rf, random_state=random_state, max_iter=max_iter
        )

    @staticmethod
    def name() -> str:
        return "missforest"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [params.Integer("n_estimators", 100, 1000, 10)]

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "MissForestPlugin":
        self._model.fit(X, *args, **kwargs)

        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self._model)

    @classmethod
    def load(cls, buff: bytes) -> "MissForestPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = MissForestPlugin
