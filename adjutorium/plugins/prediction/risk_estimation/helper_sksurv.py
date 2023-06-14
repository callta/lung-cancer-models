# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd


class sksurvWrapper:
    def __init__(self, model: Any, **kwargs: Any) -> None:
        self.model = model

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "sksurvWrapper":
        if len(args) < 2:
            raise ValueError("Invalid input for fit. Expecting X, T and Y.")

        T = args[0]
        Y = args[1]

        y = [(Y.iloc[i], T.iloc[i]) for i in range(len(Y))]
        y = np.array(y, dtype=[("status", "bool"), ("time", "<f8")])
        self.model.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if len(args) < 1:
            raise ValueError("Invalid input for predict. Expecting X and time horizon.")

        time_horizons = args[0]

        surv = self.model.predict_survival_function(X)
        preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])

        for t, eval_time in enumerate(time_horizons):
            if eval_time > np.max(surv[0].x):
                eval_time = np.max(surv[0].x)
            preds_[:, t] = np.asarray(
                [(1.0 - surv[i](eval_time)) for i in range(len(surv))]
            )

        return preds_
