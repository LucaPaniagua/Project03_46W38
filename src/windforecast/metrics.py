"""This module contains metrics for evaluating wind forecast models.
RMSE, MAE, and MSE are implemented as functions that take in two arrays
of predicted and actual values and return the corresponding metric.

"""

import numpy as np


def _check_input(y_true, y_pred):
    """Check if the inputs to arrays and check that they match in length.

    Parameters
    ----------
    y_true : array-like
        Actual observed values.
    y_pred : array-like
        Predicted values from the model.

    Returns
    -------
    y_true. y_pred : np.ndarray
        The input arrays converted to numpy arrays.

    Raises
    ------
    ValueError
        If the input arrays do not have the same length.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same length.")
    return y_true, y_pred


def rmse(y_true, y_pred):
    """Calculate the Root Mean Square Error (RMSE) between predicted and
      actual values.

    Parameters
    ----------
    y_true : array-like
        Actual observed values.
    y_pred : array-like
        Predicted values from the model.

    Returns
    -------
    float
        The RMSE value.
    """
    y_true, y_pred = _check_input(y_true, y_pred)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def mae(y_true, y_pred):
    """Calculate the Mean Absolute Error (MAE) between predicted and actual
      values.

    Parameters
    ----------
    y_true : array-like
        Actual observed values.
    y_pred : array-like
        Predicted values from the model.

    Returns
    -------
    float
        The MAE value.
    """
    y_true, y_pred = _check_input(y_true, y_pred)
    return np.mean(np.abs(y_pred - y_true))


def mse(y_true, y_pred):
    """Calculate the Mean Squared Error (MSE) between predicted
    and actual values.

    Parameters
    ----------
    y_true : array-like
        Actual observed values.
    y_pred : array-like
        Predicted values from the model.

    Returns
    -------
    float
        The MSE value.
    """
    y_true, y_pred = _check_input(y_true, y_pred)
    return np.mean((y_pred - y_true) ** 2)
