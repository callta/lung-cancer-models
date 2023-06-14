# stdlib
from typing import Union

# third party
from lifelines.utils import concordance_index as ci
import numpy as np
import pandas as pd


def _check_ndarray(arr: np.ndarray) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError("Type of arguement only supports for numpy.ndarray.")


def _check_params(model_params: Union[dict, None]) -> None:
    """
    Check `model_params`.
    """
    if model_params is None:
        raise ValueError(
            "If you want to train the model, you must \
            specify the parameter `model_params` when initializing the model."
        )

    if type(model_params) is not dict:
        raise TypeError("The type of `model_params` must be dict.")


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the concordance-index value.

    Parameters
    ----------
    y_true : np.ndarray
        Observed time. Negtive values are considered right censored.
    y_pred : np.ndarray
        Predicted value.

    Returns
    -------
    float
        Concordance index.
    """
    _check_ndarray(y_pred)
    _check_ndarray(y_true)
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    t = np.abs(y_true)
    e = (y_true > 0).astype(np.int32)
    ci_value = ci(t, y_pred, e)
    return ci_value


def _baseline_hazard(
    label_e: np.ndarray, label_t: np.ndarray, pred_hr: np.ndarray
) -> np.ndarray:
    ind_df = pd.DataFrame({"E": label_e, "T": label_t, "P": pred_hr})
    summed_over_durations = ind_df.groupby("T")[["P", "E"]].sum()
    summed_over_durations["P"] = summed_over_durations["P"].loc[::-1].cumsum()
    # where the index of base_haz is sorted time from small to large
    # and the column `base_haz` is baseline hazard rate
    base_haz = pd.DataFrame(
        summed_over_durations["E"] / summed_over_durations["P"], columns=["base_haz"]
    )
    return base_haz


def _baseline_cumulative_hazard(
    label_e: np.ndarray, label_t: np.ndarray, pred_hr: np.ndarray
) -> np.ndarray:
    return _baseline_hazard(label_e, label_t, pred_hr).cumsum()


def _baseline_survival_function(
    label_e: np.ndarray, label_t: np.ndarray, pred_hr: np.ndarray
) -> np.ndarray:
    base_cum_haz = _baseline_cumulative_hazard(label_e, label_t, pred_hr)
    survival_df = np.exp(-base_cum_haz)
    return survival_df


def baseline_survival_function(y: np.ndarray, pred_hr: np.ndarray) -> np.ndarray:
    """
    Estimate baseline survival function by Breslow Estimation.

    Parameters
    ----------
    y : np.ndarray
        Observed time. Negtive values are considered right censored.
    pred_hr : np.ndarray
        Predicted value, i.e. hazard ratio.

    Returns
    -------
    DataFrame
        Estimated baseline survival function. Index of it is time point.
        The only one column of it means the corresponding survival probability.
    """
    _check_ndarray(y)
    _check_ndarray(pred_hr)
    y = np.squeeze(y)
    pred_hr = np.squeeze(pred_hr)
    # unpack label
    t = np.abs(y)
    e = (y > 0).astype(np.int32)
    return _baseline_survival_function(e, t, pred_hr)
