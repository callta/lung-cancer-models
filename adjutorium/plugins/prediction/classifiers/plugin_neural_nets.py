# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.neural_network import MLPClassifier

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.classifiers.base as base
from adjutorium.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import adjutorium.utils.serialization as serialization


class NeuralNetsPlugin(base.ClassifierPlugin):
    """Classification plugin based on Neural networks.

    Args:
        num_layers: int
            Number of layers.
        num_units: int
            Size of the hidden layers.
        activation: str
            Activation function for the hidden layer: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        solver: str
            The solver for weight optimization: {‘lbfgs’, ‘sgd’, ‘adam’}

    Example:
        >>> from adjutorium.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("neural_nets")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    activations = ["identity", "logistic", "tanh", "relu"]
    solvers = ["lbfgs", "sgd", "adam"]

    def __init__(
        self,
        num_layers: int = 2,
        num_units: int = 100,
        activation: int = 3,
        solver: int = 2,
        calibration: int = 0,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        hidden_layer_sizes = tuple([num_units for k in range(num_layers)])
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=NeuralNetsPlugin.activations[activation],
            solver=NeuralNetsPlugin.solvers[solver],
            max_iter=10000,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "neural_nets"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("activation", 0, len(NeuralNetsPlugin.activations) - 1),
            params.Integer("solver", 0, len(NeuralNetsPlugin.solvers) - 1),
            params.Integer("num_layers", 1, 3),
            params.Integer("num_units", 50, 200),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "NeuralNetsPlugin":
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
    def load(cls, buff: bytes) -> "NeuralNetsPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = NeuralNetsPlugin
