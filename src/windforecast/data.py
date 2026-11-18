from pathlib import Path

import pandas as pd
import numpy as np


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
    
