# stdlib
from abc import ABCMeta, abstractmethod

# third party
import numpy as np
import pandas as pd


class TreatmentsPlugin(metaclass=ABCMeta):
    def __init__(self, feature_names: list = []) -> None:
        self.feature_names = feature_names

    @staticmethod
    @abstractmethod
    def name() -> str:
        ...

    @staticmethod
    def type() -> str:
        return "treatment_effects"

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame
    ) -> "TreatmentsPlugin":
        ...

    @abstractmethod
    def score(self, X: pd.DataFrame, y: pd.DataFrame, metric: str = "pehe") -> float:
        ...

    def change_output(self, output: str) -> None:
        assert output in ["pandas", "numpy"], "Invalid output type"
        if output == "pandas":
            self.output = pd.DataFrame
        elif output == "numpy":
            self.output = np.asarray
