# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.decomposition import PCA

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.preprocessors.base as base
import adjutorium.utils.serialization as serialization


class PCAPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for dimensionality reduction based on the PCA method.

    Method:
        PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Args:
        n_components: int
            Number of components to use.

    Example:
        >>> from adjutorium.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("pca")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
                    0         1
        0   -2.684126  0.319397
        1   -2.714142 -0.177001
        2   -2.888991 -0.144949
        3   -2.745343 -0.318299
        4   -2.728717  0.326755
        ..        ...       ...
        145  1.944110  0.187532
        146  1.527167 -0.375317
        147  1.764346  0.078859
        148  1.900942  0.116628
        149  1.390189 -0.282661

        [150 rows x 2 columns]
    """

    def __init__(self, model: Any = None, n_components: int = 2) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = PCA(n_components=n_components)

    @staticmethod
    def name() -> str:
        return "pca"

    @staticmethod
    def subtype() -> str:
        return "dimensionality_reduction"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        cmin, cmax = base.PreprocessorPlugin.components_interval(*args, **kwargs)
        return [params.Integer("n_components", cmin, cmax)]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "PCAPlugin":
        self.model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "PCAPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = PCAPlugin
