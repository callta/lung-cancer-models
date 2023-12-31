# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.linear_model import LogisticRegression

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.classifiers.base as base
from adjutorium.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import adjutorium.utils.serialization as serialization


class LogisticRegressionPlugin(base.ClassifierPlugin):
    """Classification plugin based on the Logistic Regression classifier.

    Method:
        Logistic regression is a linear model for classification rather than regression. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

    Args:
        C: float
            Inverse of regularization strength; must be a positive float.
        solver: str
            Algorithm to use in the optimization problem: [‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’]
        multi_class: str
            If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.
        class_weight: str
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
        max_iter: int
            Maximum number of iterations taken for the solvers to converge.

    Example:
        >>> from adjutorium.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("logistic_regression")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    #solvers = ["newton-cg", "lbfgs", "sag", "saga"]
    #classes = ["auto", "ovr", "multinomial"]
    weights = ["balanced", None]

    def __init__(
        self,
        C: float = 1.0,
        solver: str = "lbfgs",
        multi_class: str = "auto",
        class_weight: int = 1,
        max_iter: int = 100,
        penalty: str = "l2",
        calibration: int = 0,
        model: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        if solver in ['newton-cg', 'liblinear', 'lbfgs', 'sag']:
            penalty = 'l2'

        model = LogisticRegression(
            C=C,
            solver=solver,
            multi_class=multi_class,
            class_weight=LogisticRegressionPlugin.weights[class_weight],
            penalty=penalty,
            max_iter=max_iter,
            n_jobs=-1,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "logistic_regression"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("C", 1e-4, 1e4, log=True),
            params.Categorical("solver", ["newton-cg", "liblinear", "lbfgs", "sag", "saga"]),
            params.Categorical("multi_class", ["auto", "ovr", "multinomial"]),
            params.Integer("class_weight", 0, 1),
            params.Categorical("penalty", ['l2', 'l1', 'elasticnet']),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "LogisticRegressionPlugin":
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
    def load(cls, buff: bytes) -> "LogisticRegressionPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = LogisticRegressionPlugin
