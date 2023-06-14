# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.decomposition import FastICA

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.preprocessors.base as base
import adjutorium.utils.serialization as serialization


class FastICAPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for dimensionality reduction based on Independent Component Analysis algorithm.

    Method:
        Independent component analysis separates a multivariate signal into additive subcomponents that are maximally independent.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

    Args:
        n_components: int
            Number of components to use.
    Example:
        >>> from adjutorium.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("fast_ica")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
                    0         1
        0    0.112081  0.041168
        1    0.104289 -0.041024
        2    0.111794 -0.036483
        3    0.102969 -0.064518
        4    0.113980  0.042191
        ..        ...       ...
        145 -0.073602  0.039428
        146 -0.067272 -0.055427
        147 -0.068449  0.020683
        148 -0.073175  0.027519
        149 -0.060171 -0.040703

        [150 rows x 2 columns]
    """

    def __init__(self, model: Any = None, n_components: int = 2) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = FastICA(n_components=n_components, max_iter=1000)

    @staticmethod
    def name() -> str:
        return "fast_ica"

    @staticmethod
    def subtype() -> str:
        return "dimensionality_reduction"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        cmin, cmax = base.PreprocessorPlugin.components_interval(*args, **kwargs)
        return [params.Integer("n_components", cmin, cmax)]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "FastICAPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "FastICAPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = FastICAPlugin
