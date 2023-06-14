# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.cluster import FeatureAgglomeration

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.preprocessors.base as base
import adjutorium.utils.serialization as serialization


class FeatureAgglomerationPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for dimensionality reduction based on Feature Agglomeration algorithm.

    Method:
        FeatureAgglomeration uses agglomerative clustering to group together features that look very similar, thus decreasing the number of features.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html

    Args:
        n_clusters: int
            Number of clusters to find.

    Example:
        >>> from adjutorium.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("feature_agglomeration")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
                    0    1
        0    1.700000  5.1
        1    1.533333  4.9
        2    1.566667  4.7
        3    1.600000  4.6
        4    1.733333  5.0
        ..        ...  ...
        145  3.500000  6.7
        146  3.133333  6.3
        147  3.400000  6.5
        148  3.700000  6.2
        149  3.300000  5.9
    """

    def __init__(self, model: Any = None, n_clusters: int = 2) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = FeatureAgglomeration(n_clusters=n_clusters)

    @staticmethod
    def name() -> str:
        return "feature_agglomeration"

    @staticmethod
    def subtype() -> str:
        return "dimensionality_reduction"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        cmin, cmax = base.PreprocessorPlugin.components_interval(*args, **kwargs)
        return [params.Integer("n_clusters", cmin, cmax)]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "FeatureAgglomerationPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "FeatureAgglomerationPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = FeatureAgglomerationPlugin
