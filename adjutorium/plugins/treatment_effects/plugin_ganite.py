# stdlib
from typing import Optional

# third party
from ganite import Ganite
import numpy as np
import pandas as pd

# adjutorium absolute
from adjutorium.plugins.treatment_effects.base import TreatmentsPlugin
from adjutorium.utils.distributions import enable_reproducible_results
from adjutorium.utils.metrics import treatments_score

EPS = 1e-10


class GanitePlugin(TreatmentsPlugin):
    """
    The GANITE framework generates potential outcomes for a given feature vector x.
    It consists of 2 components:
     - The Counterfactual Generator block(generator + discriminator)
     - The ITE block(InferenceNets).
    """

    def __init__(
        self,
        dim_hidden: int = 40,
        alpha: float = 2,
        beta: float = 1,
        minibatch_size: int = 256,
        depth: int = 8,
        num_iterations: int = 5000,
        num_discr_iterations: int = 3,
    ) -> None:
        enable_reproducible_results()

        self.dim_hidden = dim_hidden
        self.alpha = alpha
        self.beta = beta
        self.minibatch_size = minibatch_size
        self.depth = depth
        self.num_iterations = num_iterations
        self.num_discr_iterations = num_discr_iterations

        self.model: Optional[Ganite]

    def fit(self, X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame) -> "GanitePlugin":
        X = np.asarray(X)
        T = np.asarray(T)
        Y = np.asarray(Y)

        if np.isnan(np.sum(X)):
            raise ValueError("X contains NaNs")
        if len(X) != len(T):
            raise ValueError("Features/Treatments mismatch")
        if len(X) != len(Y):
            raise ValueError("Features/Labels mismatch")

        self.model = Ganite(
            X,
            T,
            Y,
            dim_hidden=self.dim_hidden,
            alpha=self.alpha,
            beta=self.beta,
            minibatch_size=self.minibatch_size,
            depth=self.depth,
            num_iterations=self.num_iterations,
            num_discr_iterations=self.num_discr_iterations,
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call .fit first")
        return pd.DataFrame(self.model(X).detach().numpy())

    def score(self, X: pd.DataFrame, y: pd.DataFrame, metric: str = "pehe") -> float:
        y_hat = self.predict(X).to_numpy()
        return treatments_score(y, y_hat, metric)

    @staticmethod
    def name() -> str:
        return "ganite"


plugin = GanitePlugin
