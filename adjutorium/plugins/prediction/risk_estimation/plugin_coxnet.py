# stdlib
from typing import Any, List

# third party
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.risk_estimation.base as base
import adjutorium.plugins.prediction.risk_estimation.helper_sksurv as helper_sksurv
import adjutorium.utils.serialization as serialization


class CoxnetRiskEstimationPlugin(base.RiskEstimationPlugin):
    def __init__(
        self,
        n_alphas: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-7,
        model: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if model:
            self.model = model
            return

        self.model = helper_sksurv.sksurvWrapper(
            CoxnetSurvivalAnalysis(
                fit_baseline_model=True, n_alphas=n_alphas, max_iter=max_iter, tol=tol
            )
        )

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "CoxnetRiskEstimationPlugin":
        self.model.fit(X, *args, **kwargs)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "coxnet"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("n_alphas", 50, 500),
            params.Integer("max_iter", 1000, 2000),
            params.Float("tol", 1e-7, 0.1),
        ]

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "CoxnetRiskEstimationPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = CoxnetRiskEstimationPlugin
