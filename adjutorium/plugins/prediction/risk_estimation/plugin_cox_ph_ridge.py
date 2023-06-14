# stdlib
from typing import Any, List

# third party
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.risk_estimation.base as base
import adjutorium.plugins.prediction.risk_estimation.helper_sksurv as helper_sksurv
import adjutorium.utils.serialization as serialization


class CoxPHRidgePlugin(base.RiskEstimationPlugin):
    def __init__(
        self,
        alpha: float = 10.0,
        n_iter: int = 10000,
        tol: float = 1e-9,
        model: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if model:
            self.model = model
            return

        self.model = helper_sksurv.sksurvWrapper(
            CoxPHSurvivalAnalysis(alpha=alpha, n_iter=n_iter, tol=tol)
        )

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CoxPHRidgePlugin":
        self.model.fit(X, *args, **kwargs)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "cox_ph_ridge"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("alpha", 0.001, 10),
            params.Integer("n_iter", 500, 1000, 100),
            params.Float("tol", 1e-9, 0.001),
        ]

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "CoxPHRidgePlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = CoxPHRidgePlugin
