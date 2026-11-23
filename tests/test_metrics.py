"""Tests for the metrics module."""

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from windforecast.metrics import mae, mse, rmse


def test_metric_known_example():
    """Test RMSE, MAE, and MSE with a known example"""
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    assert pytest.approx(rmse(y_true, y_pred), rel=1e-9) == 0.6123724356957945
    assert pytest.approx(mae(y_true, y_pred), rel=1e-9) == 0.5
    assert pytest.approx(mse(y_true, y_pred), rel=1e-9) == 0.375


def test_metric_perfect_forecast():
    """Test RMSE, MAE, and MSE with a perfect forecast"""
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    assert rmse(y_true, y_pred) == 0.0
    assert mae(y_true, y_pred) == 0.0
    assert mse(y_true, y_pred) == 0.0


def test_metric_match_sklearn():
    """Test RMSE, MAE, and MSE against sklearn's implementation"""

    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]

    expected_mse = mean_squared_error(y_true, y_pred)
    expected_mae = mean_absolute_error(y_true, y_pred)
    expected_rmse = np.sqrt(expected_mse)

    assert mse(y_true, y_pred) == pytest.approx(expected_mse)
    assert mae(y_true, y_pred) == pytest.approx(expected_mae)
    assert rmse(y_true, y_pred) == pytest.approx(expected_rmse)


def test_metric_length_mismatch_raises():
    """Test that RMSE, MAE, and MSE raise ValueError for length mismatch"""
    y_true = [1, 2, 3]
    y_pred = [1, 2]
    with pytest.raises(ValueError):
        rmse(y_true, y_pred)
    with pytest.raises(ValueError):
        mae(y_true, y_pred)
    with pytest.raises(ValueError):
        mse(y_true, y_pred)
