# third party
from catenets.models.jax import PseudoOutcomeNet
from catenets.models.jax.transformation_utils import RA_TRANSFORMATION
import numpy as np
import pandas as pd

# adjutorium absolute
from adjutorium.plugins.treatment_effects.base import TreatmentsPlugin
from adjutorium.utils.metrics import treatments_score

EPS = 1e-10


class CATENetsPlugin(TreatmentsPlugin):
    def __init__(
        self,
        n_layers_out: int = 2,
        n_layers_r: int = 3,
        penalty_l2: float = 1e-4,
        n_iter: int = 10000,
    ) -> None:
        self.model = PseudoOutcomeNet(
            n_layers_r=n_layers_r,
            n_layers_out=n_layers_out,
            penalty_l2=penalty_l2,
            n_layers_r_t=n_layers_r,
            n_layers_out_t=n_layers_out,
            penalty_l2_t=penalty_l2,
            binary_y=True,
            transformation=RA_TRANSFORMATION,
            first_stage_strategy="S2",
            n_iter=n_iter,
        )

    def fit(
        self, X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame
    ) -> "CATENetsPlugin":
        X = np.asarray(X)
        T = np.asarray(T)
        Y = np.asarray(Y)

        if np.isnan(np.sum(X)):
            raise ValueError("X contains NaNs")
        if len(X) != len(T):
            raise ValueError("Features/Treatments mismatch")
        if len(X) != len(Y):
            raise ValueError("Features/Labels mismatch")

        self.model.fit(X=X, y=Y, w=T)

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(X))

    def score(self, X: pd.DataFrame, y: pd.DataFrame, metric: str = "pehe") -> float:
        y_hat = self.predict(X).to_numpy()
        return treatments_score(y, y_hat, metric)

    @staticmethod
    def name() -> str:
        return "catenets"


plugin = CATENetsPlugin
