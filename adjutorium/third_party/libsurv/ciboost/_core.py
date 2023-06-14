"""
Objective function and its gradients of BecCox:

L = alpha * L1 + (1 - alpha) * L2
"""
# stdlib
from typing import Tuple

# third party
import numpy as np

# adjutorium relative
from ._ci_core import _ci_grads, ci_loss
from ._efn_core import _efn_grads, efn_loss

_ALPHA = 0.0


def _params_init(params: float) -> None:
    """
    Initializer of global arguments.

    Parameters
    ----------
    params: float
        `alpha` indicates the coefficient in the objective function.

    """
    global _ALPHA

    assert params <= 1.0 and params >= 0.0
    # NOTE: If params equals 0, the first boosting round in L2 would get errors.
    #       The initial zero prediction in XGBoost results in this issue.
    if params == 0.0:
        params = 1e-2

    _ALPHA = params


def ce_evals(preds: np.ndarray, dtrain: np.ndarray) -> Tuple[str, float]:
    """
    Evaluation of BecCox model.

    Parameters
    ----------
    preds: numpy.array
        An array with shape of (N, ), where N = #data. This is also known as log hazard ratio.
    dtrain: xgboost.DMatrix
        Training data with type of `xgboost.DMatrix`. Labels can be obtained by:
        `labels = dtrain.get_label()`, and it is `numpy.array` with shape of (N, ), where N = #data.

    Returns
    -------
    float:
        Concordance index.
    """
    N = preds.shape[0]
    # labels
    labels = dtrain.get_label().astype("int")
    E = (labels > 0).astype("int")
    T = np.abs(labels)

    # count indicator
    nsum = 0
    ny = 0
    for i in range(N):
        if E[i] > 0:
            p = preds[i] - preds[T[i] < T]
            nsum += p.shape[0]
            ny += np.sum(p > 0)

    CI = 1.0 * ny / nsum

    return "ce_evals", CI


def ce_loss(preds: np.ndarray, dtrain: np.ndarray) -> Tuple[str, float]:
    """
    Computation of Objective Function.
    L = alpha * L1 + (1 - alpha) * L2.

    Parameters
    ----------
    preds: numpy.array
        An array with shape of (N, ), where N = #data. This is also known as log hazard ratio.
    dtrain: xgboost.DMatrix
        Training data with type of `xgboost.DMatrix`. Labels can be obtained by:
        `labels = dtrain.get_label()`, and it is `numpy.array` with shape of (N, ), where N = #data.

    Returns
    -------
    tuple:
        Name and value of objective function defined in BecCox model.

    Notes
    -----
    Absolute value of label represents `T` in survival data, Negtive values are considered
    right censored, i.e. `E = 0`; Positive values are considered event occurrence, i.e. `E = 1`.
    """
    __, L1_loss = efn_loss(preds, dtrain)
    __, L2_loss = ci_loss(preds, dtrain)
    return "ce_loss", _ALPHA * L1_loss + (1.0 - _ALPHA) * L2_loss


def _ce_grads(preds: np.ndarray, dtrain: np.ndarray) -> Tuple[float, float]:
    """
    Gradient computation of custom objective function.

    Parameters
    ----------
    preds: numpy.array
        An array with shape of (N, ), where N = #data. This is also known as log hazard ratio.
    dtrain: xgboost.DMatrix
        Training data with type of `xgboost.DMatrix`. Labels can be obtained by:
        `labels = dtrain.get_label()`, and it is `numpy.array` with shape of (N, ), where N = #data.

    Returns
    -------
    tuple:
        The first- and second-order gradients of objective function w.r.t. `preds`.
    """
    L1_grads, L1_hess = _efn_grads(preds, dtrain)
    L2_grads, L2_hess = _ci_grads(preds, dtrain)
    return (
        _ALPHA * L1_grads + (1.0 - _ALPHA) * L2_grads,
        _ALPHA * L1_hess + (1.0 - _ALPHA) * L2_hess,
    )
