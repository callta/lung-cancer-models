# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
from xgboost import DMatrix

# adjutorium absolute
from adjutorium.third_party.libsurv.datasets import survival_dmat


def factorize_categorical(df: pd.DataFrame) -> pd.DataFrame:
    categorical = df.select_dtypes(include=["category"]).columns
    for col in categorical:
        df[col] = pd.factorize(df[col])[0]

    return df


class libsurvWrapper:
    def __init__(self, model: Any, **kwargs: Any) -> None:
        self.model = model

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "libsurvWrapper":
        if len(args) < 2:
            raise ValueError("Invalid input for fit. Expecting X, T and Y.")

        T = args[0]
        Y = args[1]

        y = pd.DataFrame({"t": T, "e": Y})

        data_train = X.join(y)
        data_train = factorize_categorical(data_train)

        surv_train = survival_dmat(data_train, t_col="t", e_col="e", label_col="Y")
        self.model.train(
            surv_train,
            num_rounds=200,
            skip_rounds=10,
            silent=True,
            evals=[(surv_train, "train")],
        )

        return self

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if len(args) < 1:
            raise ValueError("Invalid input for predict. Expecting X and time horizon.")

        time_horizons = args[0]

        X = factorize_categorical(X)
        X = DMatrix(X)

        surv = self.model.predict_survival_function(X)
        surv_times = np.asarray(surv.T.index).astype(int)
        surv = np.asarray(surv)
        preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])

        for t, eval_time in enumerate(time_horizons):
            tmp_time = np.where(eval_time <= surv_times)[0]
            if len(tmp_time) == 0:
                preds_[:, t] = 1.0 - surv[:, 0]
            else:
                preds_[:, t] = 1.0 - surv[:, tmp_time[0]]

        return preds_
