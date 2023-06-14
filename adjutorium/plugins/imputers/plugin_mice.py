"""
"""
# stdlib
import time
from typing import Any, List, Union

# third party
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.imputers.base as base
import adjutorium.plugins.utils.decorators as decorators
import adjutorium.utils.serialization as serialization


class MicePlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Multivariate Iterative chained equations and multiple imputations.

    Method:
        Multivariate Iterative chained equations(MICE) methods model each feature with missing values as a function of other features in a round-robin fashion. For each step of the round-robin imputation, we use a BayesianRidge estimator, which does a regularized linear regression.
        The class `sklearn.impute.IterativeImputer` is able to generate multiple imputations of the same incomplete dataset. We can then learn a regression or classification model on different imputations of the same dataset.
        Setting `sample_posterior=True` for the IterativeImputer will randomly draw values to fill each missing value from the Gaussian posterior of the predictions. If each `IterativeImputer` uses a different `random_state`, this results in multiple imputations, each of which can be used to train a predictive model.
        The final result is the average of all the `n_imputation` estimates.

    Args:
        n_imputations: int, default=5i
            number of multiple imputations to perform.
        max_iter: int, default=500
            maximum number of imputation rounds to perform.
        random_state: int, default set to the current time.
            seed of the pseudo random number generator to use.

    Example:
        >>> import numpy as np
        >>> from adjutorium.plugins.imputers import Imputers
        >>> plugin = Imputers().get("mice")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
                  0        1         2         3
        0  1.000000  1.00000  1.000000  1.000000
        1  1.222412  1.68686  1.687483  1.221473
        2  1.000000  2.00000  2.000000  1.000000
        3  2.000000  2.00000  2.000000  2.000000
    """

    def __init__(
        self,
        n_imputations: int = 1,
        max_iter: int = 100,
        random_state: Union[int, None] = None,
        model: Any = None,
    ) -> None:
        super().__init__()

        if model:
            self._models = model
            return

        if not random_state:
            random_state = int(time.time())

        self._models = []
        for idx in range(n_imputations):
            self._models.append(
                IterativeImputer(
                    max_iter=max_iter,
                    sample_posterior=True,
                    random_state=random_state + idx,
                )
            )

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "MicePlugin":
        for model in self._models:
            model.fit(X, *args, **kwargs)

        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        imputations = []
        for model in self._models:
            X_reconstructed = model.transform(X)
            imputations.append(X_reconstructed)

        return np.mean(imputations, axis=0)

    @staticmethod
    def name() -> str:
        return "mice"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [params.Integer("n_imputations", 1, 3)]

    def save(self) -> bytes:
        return serialization.save_model(self._models)

    @classmethod
    def load(cls, buff: bytes) -> "MicePlugin":
        models = serialization.load_model(buff)
        return cls(model=models)


plugin = MicePlugin
