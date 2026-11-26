"""Tests for the models module."""

import pandas as pd
import pytest

from windforecast.models import ForecastModel, PersistenceModel


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
