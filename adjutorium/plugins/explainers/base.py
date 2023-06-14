# stdlib
from abc import ABCMeta, abstractmethod

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ExplainerPlugin(metaclass=ABCMeta):
    def __init__(self, feature_names: list = []) -> None:
        self.feature_names = feature_names

    @staticmethod
    @abstractmethod
    def name() -> str:
        ...

    @staticmethod
    @abstractmethod
    def pretty_name() -> str:
        ...

    @staticmethod
    def type() -> str:
        return "explainer"

    @abstractmethod
    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def plot(
        self,
        importances: pd.DataFrame,
    ) -> None:

        importances = np.asarray(importances)

        title = f"{self.name()} importance"
        axis_title = "Features"

        x_pos = np.arange(len(self.feature_names))

        plt.figure(figsize=(12, 6))
        plt.bar(x_pos, importances, align="center")
        plt.xticks(x_pos, self.feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
        plt.show()
