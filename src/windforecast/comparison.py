"""Comparison is the file to used automatice the models working with the same
values and compare de results with that"""

import pandas as pd

from windforecast.metrics import mae, mse, rmse


class ModelComparison:
    """ "Model comparison it a class that compare the different models between
    and help you to brings an idea what it's precise"""

    def __init__(
        self,
        site,
        models,
        test_start="2021-01-01",
        test_end="2021-12-31",
        horizon=1,
    ):
        """Docstring"""
        self.site = site
        self.models = models
        self.X_train, self.X_test, self.y_train, self.y_test = (
            site.train_test_split(test_start, test_end, horizon)
        )
        self.prediction = {}
        self.results = None

    def run(self):
        """Train every model, forecast the test period and compare them

        Returns
        ----------
        pd.Dataframe
            One row per model with its MSE, MAE and RMSE

        """

        rows = []
        for model in self.models:
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            self.prediction[model.name] = y_pred
            rows.append(
                {
                    "model": model.name,
                    "MSE": mse(self.y_test, y_pred),
                    "MAE": mae(self.y_test, y_pred),
                    "RMSE": rmse(self.y_test, y_pred),
                }
            )
        self.results = pd.DataFrame(rows).set_index("model")
        return self.results
