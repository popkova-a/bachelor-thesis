import warnings
import numpy as np
from typing import Any, Iterable, Union
warnings.filterwarnings("ignore", category=RuntimeWarning)


def empirical_error(y_true: Iterable[float], y_pred: Iterable[float]) -> Union[float, np.ndarray]:
    """
    Computes the empirical error between the ground truth values
    and the predicted ones.

    :param y_true:  Iterable[float],                ground truth values
    :param y_pred:  Iterable[float],                predicted values
    :return:        Union[float, np.ndarray],       empirical error
    """

    if isinstance(y_true, Iterable) and isinstance(y_pred, Iterable):
        try:
            y_true = np.array(y_true, dtype=float)
            y_pred = np.array(y_pred, dtype=float)
        except BaseException:
            raise TypeError("The input must be of the iterable type over floats.")
    else:
        raise TypeError("The input must be of the iterable type over floats.")

    if (len(y_true) == 0) or (len(y_pred) == 0):
        raise ValueError("The input is empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("The inputs must be of the same size.")

    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("The input contains NaN.")

    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("The input contains Inf.")

    return np.mean((y_true - y_pred) ** 2, axis=0)


def r_squared(y_true: Iterable[float], y_pred: Iterable[float]) -> Union[float, np.ndarray]:
    """
    Computes the R^2 regression score.

    :param y_true:  Iterable[float],                ground truth values
    :param y_pred:  Iterable[float],                predicted values
    :return:        Union[float, np.ndarray],       R^2 score
    """

    if isinstance(y_true, Iterable) and isinstance(y_pred, Iterable):
        try:
            y_true = np.array(y_true, dtype=float)
            y_pred = np.array(y_pred, dtype=float)
        except BaseException:
            raise TypeError("The input must be of the iterable type over floats.")
    else:
        raise TypeError("The input must be of the iterable type over floats.")

    if (len(y_true) == 0) or (len(y_pred) == 0):
        raise ValueError("The input is empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("The inputs must be of the same size.")

    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("The input contains NaN.")

    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("The input contains Inf.")

    if len(y_true) == 1:
        raise RuntimeError("R^2 score is not well-defined with less than two samples.")

    e_emp = empirical_error(y_true, y_pred)
    var = np.sum((y_true - np.mean(y_true)) ** 2, axis=0) / (len(y_true) - 1)

    # If y_true is constant, then the R^2 score isn't finite: it is either NaN
    # (perfect predictions) or -Inf (imperfect predictions). To prevent
    # such non-finite numbers, y default these cases are replaced with 1.0
    # (perfect predictions) or 0.0 (imperfect predictions) respectively.

    res = 1 - e_emp / var
    nan_res = np.isnan(res)
    inf_res = np.isinf(res)

    if np.sum(nan_res) > 0:
        if isinstance(res, (int, float)):
            return 1.
        else:
            res[nan_res] = 1.

    if np.sum(inf_res) > 0:
        if isinstance(res, (int, float)):
            return 0.
        else:
            res[inf_res] = 0.

    return res


def mean_absolute_error(y_true: Iterable[float], y_pred: Iterable[float]) -> Union[float, np.ndarray]:
    """
    Computes the Mean Absolute Error (MAE) between the ground
    truth values and the predicted ones.

    :param y_true:  Iterable[float],                ground truth values
    :param y_pred:  Iterable[float],                predicted values
    :return:        Union[float, np.ndarray],       Mean Absolute Error
    """

    if isinstance(y_true, Iterable) and isinstance(y_pred, Iterable):
        try:
            y_true = np.array(y_true, dtype=float)
            y_pred = np.array(y_pred, dtype=float)
        except BaseException:
            raise TypeError("The input must be of the iterable type over floats.")
    else:
        raise TypeError("The input must be of the iterable type over floats.")

    if (len(y_true) == 0) or (len(y_pred) == 0):
        raise ValueError("The input is empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("The inputs must be of the same size.")

    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("The input contains NaN.")

    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("The input contains Inf.")

    return np.mean(np.abs(y_true - y_pred), axis=0)


def relative_mean_absolute_error(y_true: Iterable[float], y_pred: Iterable[float]) -> Union[float, np.ndarray]:
    """
    Computes the relative Mean Absolute Error (rMAE) between the ground
    truth values and the predicted ones.

    :param y_true:  Iterable[float],                ground truth values
    :param y_pred:  Iterable[float],                predicted values
    :return:        Union[float, np.ndarray],       relative Mean Absolute Error
    """

    if isinstance(y_true, Iterable) and isinstance(y_pred, Iterable):
        try:
            y_true = np.array(y_true, dtype=float)
            y_pred = np.array(y_pred, dtype=float)
        except BaseException:
            raise TypeError("The input must be of the iterable type over floats.")
    else:
        raise TypeError("The input must be of the iterable type over floats.")

    if (len(y_true) == 0) or (len(y_pred) == 0):
        raise ValueError("The input is empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("The inputs must be of the same size.")

    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("The input contains NaN.")

    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("The input contains Inf.")

    if np.isclose(y_true, 0.).any():
        raise ValueError("rMAE is not well-defined when any ground truth value equals zero.")

    return np.mean(np.abs(1 - y_pred/y_true), axis=0)


def leave_one_out_error(sample: Iterable[float], basis: Any, y_true: Iterable[float],
                        y_pred: Iterable[float]) -> Union[float, np.ndarray]:
    """
    Computes the leave-one-out error between the ground truth values and
    the predicted ones using an information matrix of the PC expansion.

    :param sample:  Iterable[float],                experimental design samples
    :param basis:   Any,                            PC basis
    :param y_true:  Iterable[float],                ground truth values
    :param y_pred:  Iterable[float],                predicted values
    :return:        Union[float, np.ndarray],       leave-one-out error
    """

    if isinstance(y_true, Iterable) and isinstance(y_pred, Iterable) and\
       isinstance(sample, Iterable):
        try:
            sample = np.array(sample, dtype=float)
            y_true = np.array(y_true, dtype=float)
            y_pred = np.array(y_pred, dtype=float)
        except BaseException:
            raise TypeError("The input must be of the iterable type over floats.")
    else:
        raise TypeError("The input must be of the iterable type over floats.")

    if (len(sample) == 0) or (len(basis) == 0) or\
       (len(y_true) == 0) or (len(y_pred) == 0):
        raise ValueError("The input is empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("The inputs must be of the same size.")

    if np.isnan(sample).any() or np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("The input contains NaN.")

    if np.isinf(sample).any() or np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("The input contains Inf.")

    psi_mat = basis(sample).T
    h_diag = np.diag(psi_mat @ np.linalg.inv(psi_mat.T @ psi_mat) @ psi_mat.T)

    if np.sum(np.isinf(h_diag) | np.isnan(h_diag) | np.isclose(h_diag, 1.)) > 0:
        raise RuntimeError("The diagonal of the regression matrix contains either Inf, NaN or ones.")

    return np.mean((y_true - y_pred) ** 2 / (1 - h_diag.reshape(-1, 1)) ** 2, axis=0)


def relative_leave_one_out_error(sample: Iterable[float], basis: Any, y_true: Iterable[float],
                                 y_pred: Iterable[float]) -> Union[float, np.ndarray]:
    """
    Computes the relative leave-one-out error between the ground truth values and
    the predicted ones using an information matrix of the PC expansion.

    :param sample:  Iterable[float],                experimental design samples
    :param basis:   Any,                            PC basis
    :param y_true:  Iterable[float],                ground truth values
    :param y_pred:  Iterable[float],                predicted values
    :return:        Union[float, np.ndarray],       relative leave-one-out error
    """

    if isinstance(y_true, Iterable) and isinstance(y_pred, Iterable) and \
            isinstance(sample, Iterable):
        try:
            sample = np.array(sample, dtype=float)
            y_true = np.array(y_true, dtype=float)
            y_pred = np.array(y_pred, dtype=float)
        except BaseException:
            raise TypeError("The input must be of the iterable type over floats.")
    else:
        raise TypeError("The input must be of the iterable type over floats.")

    if (len(sample) == 0) or (len(basis) == 0) or \
            (len(y_true) == 0) or (len(y_pred) == 0):
        raise ValueError("The input is empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("The inputs must be of the same size.")

    if np.isnan(sample).any() or np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("The input contains NaN.")

    if np.isinf(sample).any() or np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("The input contains Inf.")

    if len(y_true) == 1:
        raise RuntimeError("Relative E_LOO score is not well-defined with less than two samples.")

    e_loo = leave_one_out_error(sample, basis, y_true, y_pred)
    var = np.sum((y_true - np.mean(y_true)) ** 2, axis=0) / (len(y_true) - 1)

    # If y_true is constant, then the score isn't finite: it is either NaN
    # (perfect predictions) or -Inf (imperfect predictions). To prevent
    # such non-finite numbers, y default these cases are replaced with 1.0
    # (perfect predictions) or 0.0 (imperfect predictions) respectively.

    res = e_loo / var
    nan_res = np.isnan(res)
    inf_res = np.isinf(res)

    if np.sum(nan_res) > 0:
        if isinstance(res, (int, float)):
            return 1.
        else:
            res[nan_res] = 1.

    if np.sum(inf_res) > 0:
        if isinstance(res, (int, float)):
            return 0.
        else:
            res[inf_res] = 0.

    return res


def q_squared(sample: Iterable[float], basis: Any, y_true: Iterable[float],
              y_pred: Iterable[float]) -> Union[float, np.ndarray]:
    """
    Computes the Q^2 score.

    :param sample:  Iterable[float],                experimental design samples
    :param basis:   Any,                            PC basis
    :param y_true:  Iterable[float],                ground truth values
    :param y_pred:  Iterable[float],                predicted values
    :return:        Union[float, np.ndarray],       Q^2 score
    """

    return 1 - relative_leave_one_out_error(sample, basis, y_true, y_pred)
