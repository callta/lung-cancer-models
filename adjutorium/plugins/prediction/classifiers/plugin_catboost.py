# stdlib
from typing import Any, List

# third party
from catboost import CatBoostClassifier
import pandas as pd

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.classifiers.base as base
from adjutorium.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import adjutorium.utils.serialization as serialization


class CatBoostPlugin(base.ClassifierPlugin):
    """Classification plugin based on the CatBoost framework.

    Method:
        CatBoost provides a gradient boosting framework which attempts to solve for Categorical features using a permutation driven alternative compared to the classical algorithm. It uses Ordered Boosting to overcome over fitting and Symmetric Trees for faster execution.

    Args:
        learning_rate: float
            The learning rate used for training.
        depth: int

        iterations: int

        od_type: int

        od_wait: int

        border_count: int

        l2_leaf_reg: float

        random_strength: float

        grow_policy: int

    Example:
        >>> from adjutorium.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("catboost")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    grow_policies = ["Depthwise", "SymmetricTree", "Lossguide"]

    def __init__(
        self,
        learning_rate: float = 1e-2,
        depth: int = 6,
        iterations: int = 500,
        od_type: str = "Iter",
        od_wait: int = 100,
        border_count: int = 128,
        l2_leaf_reg: float = 1e-4,
        random_strength: float = 0,
        grow_policy: int = 0,
        calibration: int = 0,
        model: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = CatBoostClassifier(
            depth=depth,
            logging_level="Silent",
            allow_writing_files=False,
            used_ram_limit="6gb",
            iterations=iterations,
            od_type=od_type,
            od_wait=od_wait,
            border_count=border_count,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            grow_policy=CatBoostPlugin.grow_policies[grow_policy],
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "catboost"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("learning_rate", 1e-2, 4e-2),
            params.Integer("depth", 4, 12),
            params.Float("l2_leaf_reg", 1e-4, 1e3),
            params.Float("random_strength", 0, 3),
            params.Integer("grow_policy", 0, len(CatBoostPlugin.grow_policies) - 1),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CatBoostPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "CatBoostPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = CatBoostPlugin
