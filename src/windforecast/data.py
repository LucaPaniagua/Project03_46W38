from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


class WindSite:
    """One wind site with it's hourly wind speed data and power data

    Parameters
    ----------
    site_index: int
        Site number, from 1 to 4. it's selects the site from the data file.
        inputs/Location<site_index>.csv
    input_dir: str
        Path to the input directory where the data files are located.

    """

    VALID_SITE_INDEX = [1, 2, 3, 4]

    VARIABLE_UNITS = {
        "temperature_2m": "°F",
        "relativehumidity_2m": "%",
        "dewpoint_2m": "°F",
        "windspeed_10m": "m/s",
        "windspeed_100m": "m/s",
        "winddirection_10m": "°",
        "winddirection_100m": "°",
        "windgusts_10m": "m/s",
        "Power": "-",
    }

    def __init__(self, site_index: int, input_dir="inputs"):
        """Initialize the WindSite object with the site index and input
        directory"""

        if site_index not in self.VALID_SITE_INDEX:
            raise ValueError(
                f"site_index must be one of {self.VALID_SITE_INDEX}, "
                f"got {site_index}"
            )
        self.site_index = site_index
        self.path = Path(input_dir) / f"Location{site_index}.csv"
        self.data = self._load_data()

    def _load_data(self):
        """Read the CSV file with the time column as a datetime index"""
        return pd.read_csv(self.path, parse_dates=["Time"], index_col="Time")

    def __repr__(self):
        return f"WindSite(site_index={self.site_index}, path='{self.path}')"

    def __str__(self):
        return f"WindSite {self.site_index} with data from {self.path}"

    def plot_variable(self, variable_name, start, end):
        """Plot a variable from the data between start and end dates

        Parameters
        ----------
        variable_name : str
            Name of the variable to plot
        start : pd.Timestamp
            Start date for the plot
        end : pd.Timestamp
            End date for the plot

        """

        if variable_name not in self.data.columns:
            raise ValueError(
                f"Variable '{variable_name}' not found"
                f" in data columns: {self.data.columns.tolist()}"
            )

        plot_data = self.data[variable_name].loc[start:end]
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_data.plot(ax=ax)
        if plot_data.empty:
            raise ValueError(
                f"No data available for the specified range:"
                f" {start} to {end}"
            )
        ax.set_xlabel("Time")
        ax.set_ylabel(
            f"{variable_name}"
            f"[{self.VARIABLE_UNITS.get(variable_name, '')}]"
        )
        ax.set_title(f"{variable_name} for WindSite {self.site_index}")
        return fig, ax

    def make_features(self, horizon=1):
        """Build the features X(t) and the target y = Power(t + horizon).

        Parameters
        ----------
        horizon : int, default=1
            Number of hours ahead to forecast.

        Returns
        -------
        X : pd.DataFrame
            Features available at time t.
        y : pd.Series
            Power at time t + horizon.
        """
        columns = [
            "windspeed_10m",
            "windspeed_100m",
            "Power",
        ]
        X = self.data[columns].copy()
        sin_dir_100m, cos_dir_100m = _deg_to_sin_cos(
            self.data["winddirection_100m"]
        )
        X["sin_winddirection_100m"] = sin_dir_100m
        X["cos_winddirection_100m"] = cos_dir_100m

        y = self.data["Power"].shift(-horizon)
        valid = y.notna()
        return X[valid], y[valid]

    def train_test_split(
        self, test_start="2021-01-01", test_end="2021-12-31", horizon=1
    ):
        """Split the data into training and testing sets based on the provided
        date of starr and end date.

        Parameters
        ----------
        test_start : str
            Start date for the test set in 'YYYY-MM-DD' format.
        test_end : str
            End date for the test set in 'YYYY-MM-DD' format.

        Returns
        -------
        X_train : pd.DataFrame
            Features for the training set.
        y_train : pd.Series
            Target for the training set.
        X_test : pd.DataFrame
            Features for the testing set.
        y_test : pd.Series
            Target for the testing set.
        """

        X, y = self.make_features(horizon=horizon)
        X_train = X[X.index < test_start]
        y_train = y[y.index < test_start]
        X_test = X.loc[test_start:test_end]
        y_test = y.loc[test_start:test_end]

        return X_train, X_test, y_train, y_test


def _deg_to_sin_cos(winddirection):
    """convert wind direction from degrees to sine and cosine values

    Parameters
    ----------
    winddirection : pd.Series
        wind direction in degrees

    Returns
    -------
    sin_direction : float or pd.series
        sine of wind direction
    cos_direction : float or pd.series
        cosine of wind direction

    """

    winddirection_rad = np.deg2rad(winddirection)
    return np.sin(winddirection_rad), np.cos(winddirection_rad)
