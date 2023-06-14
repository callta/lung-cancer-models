# stdlib
from typing import Optional

# third party
from cmgp import CMGP
import pandas as pd

# adjutorium absolute
from adjutorium.plugins.treatment_effects.base import TreatmentsPlugin
from adjutorium.utils.metrics import treatments_score


class CMGPPlugin(TreatmentsPlugin):
    def __init__(
        self,
        mode: str = "CMGP",
        max_gp_iterations: int = 1000,
    ) -> None:
        self.mode = mode
        self.max_gp_iterations = max_gp_iterations

        self.model: Optional[CMGP]

    def fit(self, X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame) -> "CMGPPlugin":
        self.model = CMGP(
            X, T, Y, mode=self.mode, max_gp_iterations=self.max_gp_iterations
        )

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call .fit first")
        return pd.DataFrame(self.model.predict(X))

    def score(self, X: pd.DataFrame, y: pd.DataFrame, metric: str = "pehe") -> float:
        y_hat = self.predict(X).to_numpy()
        return treatments_score(y, y_hat, metric)

    @staticmethod
    def name() -> str:
        return "cmgp"


plugin = CMGPPlugin
