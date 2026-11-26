"""This File contains the model"""

from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class ForecastModel(ABC):
    """Evary type of forecasting model need to follow"""

    name = "base"

    def fit(self, X_train, y_train):
        """Train the model on the training data.

        By default nothing is learned, so models that need no training
        (like persistence) can inherit this method. Subclasses that learn
        from data override it.

        Parameters
        ----------
        X_train : pd.DataFrame
            Features of the training period.
        y_train : pd.Series
            Target of the training period.

        Returns
        -------
        ForecastModel
            The model itself, so calls can be chained.
        """

        return self

    @abstractmethod
    def predict(self, X):
        """Return the forecast of the power for each row of X.

        Every subclass must implement this method.
        """


class PersistenceModel(ForecastModel):
    """Baseline model: the power of the next hour equals the current one.

    It needs no training, so it inherits the default fit. It is the
    reference that the machine learning models must outperform.
    """

    name = "Persistence"

    def predict(self, X):
        """Forecast the next hour as the current power.

        Parameters
        ----------
        X : pd.DataFrame
            Features from WindSite.make_features. The column "Power"
            holds the power at time t.

        Returns
        -------
        np.ndarray
            The forecast of the power at time t + horizon.
        """

        return X["Power"].to_numpy()


class LinearRegressionModel(ForecastModel):
    name = "Linear regression"

    def __init__(self):
        self.estimator = LinearRegression()

    def fit(self, X_train, y_train):
        self.estimator.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.estimator.predict(X)


class RandomForestModel(ForecastModel):
    name = "Random forest"

    def __init__(self):
        self.estimator = RandomForestRegressor(random_state=0, n_jobs=-1)

    def fit(self, X_train, y_train):
        self.estimator.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.estimator.predict(X)
