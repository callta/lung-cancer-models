# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.preprocessors.base as base
import adjutorium.utils.serialization as serialization


class InteractionsPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for adding interactions.

    Method:
        This method transforms the features to generate interaction terms.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html?highlight=polynomial#sklearn.preprocessing.PolynomialFeatures.fit

    Example:
        >>> from adjutorium.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("interactions")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(self, model: Any = None) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = PolynomialFeatures(interaction_only=True)

    @staticmethod
    def name() -> str:
        return "interactions"

    @staticmethod
    def subtype() -> str:
        return "feature_scaling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "InteractionsPlugin":

        self.model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "InteractionsPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = InteractionsPlugin
