"""Tests for the models module."""

import pandas as pd
import pytest

from windforecast.models import (
    ForecastModel,
    PersistenceModel,
    LinearRegressionModel,
    RandomForestModel,
)


def test_forecast_model_is_abstract():
    """ForecastModel is a contract and cannot be instantiated."""
    with pytest.raises(TypeError):
        ForecastModel()


def test_persistence_predicts_current_power():
    """Persistence forecasts the next hour as the current power."""
    X = pd.DataFrame({"Power": [0.1, 0.5, 0.9]})
    model = PersistenceModel()
    assert list(model.predict(X)) == [0.1, 0.5, 0.9]


def test_fit_returns_self():
    """fit returns the model itself, so calls can be chained."""
    X = pd.DataFrame({"Power": [0.1, 0.5, 0.9]})
    y = pd.Series([0.5, 0.9, 0.3])
    model = PersistenceModel()
    assert model.fit(X, y) is model


def test_linear_regression_leans_relation():
    """Test if linear regresion recovers an exact linear correlation"""
    X = pd.DataFrame({"Power": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 3.0, 5.0, 7.0])  # y = 2 * Power + 1
    model = LinearRegressionModel().fit(X, y)
    assert list(model.predict(X)) == pytest.approx([1.0, 3.0, 5.0, 7.0])


def test_random_forest_is_reproductible():
    """Test two random forest trained with the same data
    and compare the results of the forecast"""
    X = pd.DataFrame({"Power": [0.0, 1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0.4, 0.2, 0.8, 0.6, 0.5])
    rand_forest_model_test_1 = RandomForestModel().fit(X, y).predict(X)
    rand_forest_model_test_2 = RandomForestModel().fit(X, y).predict(X)
    assert list(rand_forest_model_test_1) == pytest.approx(
        list(rand_forest_model_test_2)
    )


def test_ml_models_fit_returns_self():
    """fit returns the model for both machine learning models"""
    X = pd.DataFrame({"Power": [0.1, 0.5, 0.9]})
    y = pd.Series([0.5, 0.9, 0.3])
    test_models = [LinearRegressionModel(), RandomForestModel()]
    for model in test_models:
        assert model.fit(X, y) is model
