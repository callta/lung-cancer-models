# stdlib
from typing import Any, List

# third party
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.risk_estimation.base as base
import adjutorium.plugins.prediction.risk_estimation.helper_sksurv as helper_sksurv
import adjutorium.utils.serialization as serialization


class RandomSurvivalForestPlugin(base.RiskEstimationPlugin):
    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        min_weight_fraction_leaf: float = 0,
        model: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if model:
            self.model = model
            return

        self.model = helper_sksurv.sksurvWrapper(
            RandomSurvivalForest(
                n_estimators=n_estimators,
                max_depth=5,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                n_jobs=2,
            )
        )

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "RandomSurvivalForestPlugin":
        self.model.fit(X, *args, **kwargs)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "random_survival_forest"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("n_estimators", 50, 400, 50),
            params.Integer("min_samples_split", 2, 50),
            params.Integer("min_samples_leaf", 2, 30),
            params.Float("min_weight_fraction_leaf", 0, 0.5),
        ]

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "RandomSurvivalForestPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = RandomSurvivalForestPlugin
