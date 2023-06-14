# stdlib
from typing import Any, List

# third party
import pandas as pd
from xgboost import XGBClassifier

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.classifiers.base as base
from adjutorium.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import adjutorium.utils.serialization as serialization


class XGBoostPlugin(base.ClassifierPlugin):
    """Classification plugin based on the XGBoost classifier.

    Method:
        Gradient boosting is a supervised learning algorithm that attempts to accurately predict a target variable by combining an ensemble of estimates from a set of simpler and weaker models. The XGBoost algorithm has a robust handling of a variety of data types, relationships, distributions, and the variety of hyperparameters that you can fine-tune.

    Args:
        n_estimators: int
            The maximum number of estimators at which boosting is terminated.
        max_depth: int
            Maximum depth of a tree.
        reg_lambda: float
            L2 regularization term on weights (xgb’s lambda).
            Increasing this value will make model more conservative.
        reg_alpha: float
            L1 regularization term on weights (xgb’s alpha).
            Increasing this value will make model more conservative.
        colsample_bytree: float
            Subsample ratio of columns when constructing each tree.
        colsample_bynode: float
             Subsample ratio of columns for each split.
        colsample_bylevel: float
             Subsample ratio of columns for each level.
        subsample: float
            Subsample ratio of the training instance.
        learning_rate: float
            Boosting learning rate; alias of eta
            Step size shrinkage used in update to prevents overfitting. 
            After each boosting step, we can directly get the weights of new features, 
            and eta shrinks the feature weights to make the boosting process more conservative.
        gamma: float
            Minimum loss reduction required to 
            make a further partition on a leaf node of the tree.
            The larger gamma is, the more conservative the algorithm will be.
        booster: str
            Specify which booster to use: gbtree, gblinear or dart.
        min_child_weight: int
            Minimum sum of instance weight(hessian) needed in a child.
            The larger min_child_weight is, the more conservative the algorithm will be.
        max_bin: int
            Number of bins for histogram construction.
        tree_method: str
            Specify which tree method to use. Default to auto. 
            If this parameter is set to default, XGBoost will 
            choose the most conservative option available.
        random_state: float
            Random number seed.


    Example:
        >>> from adjutorium.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("xgboost")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    booster = ["gbtree", "gblinear", "dart"]

    def __init__(
        self,
        n_estimators: int = 100,
        reg_lambda: float = 1,
        reg_alpha: float = 0,
        colsample_bytree: float = 1,
        colsample_bynode: float = 1,
        colsample_bylevel: float = 1,
        max_depth: int = 6,
        subsample: float = 0.1,
        learning_rate: float = 0.3,
        gamma: float=0,
        min_child_weight: int = 1,
        max_bin: int = 256,
        tree_method: str = "hist",
        grow_policy: str = "depthwise",
        booster: int = 0,
        random_state: int = 0,
        calibration: int = 0,
        model: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = XGBClassifier(
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
            colsample_bylevel=colsample_bylevel,
            max_depth=max_depth,
            subsample=subsample,
            learning_rate=learning_rate,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_bin=max_bin,
            verbosity=0,
            use_label_encoder=False,
            tree_method=tree_method,
            grow_policy=grow_policy,
            booster=XGBoostPlugin.booster[booster],
            random_state=random_state,
            n_estimators=n_estimators,
            nthread=2,
            **kwargs,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "xgboost"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("reg_lambda", 1e-8, 1.0, log=True),
            params.Float("reg_alpha", 1e-8, 1.0, log=True),
            params.Float("colsample_bytree", 0.1, 0.9),
            params.Float("colsample_bynode", 0.1, 0.9),
            params.Float("colsample_bylevel", 0.1, 0.9),
            params.Float("subsample", 0.1, 0.9),
            params.Float("learning_rate", 0.3, 1.0, log=True),
            params.Float("gamma", 1e-8, 1.0, log=True),
            params.Integer("max_depth", 2, 9),
            params.Integer("min_child_weight", 0, 300),
            params.Integer("max_bin", 256, 512),
            params.Integer("booster", 0, len(XGBoostPlugin.booster) - 1),
            params.Categorical("grow_policy", ["depthwise", "lossguide"])
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "XGBoostPlugin":
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
    def load(cls, buff: bytes) -> "XGBoostPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = XGBoostPlugin
