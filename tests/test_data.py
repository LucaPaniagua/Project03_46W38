import matplotlib.pyplot as plt
import pytest

from windforecast.data import WindSite, _deg_to_sin_cos


def test_wind_site_shape_and_index():
    """Site 1 load 43800 hourly row with Time as index"""

    site = WindSite(1)
    assert site.data.shape == (43800, 9)
    assert site.data.index.name == "Time"


def test_invalid_site_raises():
    """A site number outside 1-4 and gives a ValueError"""
    with pytest.raises(ValueError):
        WindSite(5)


def test_deg_to_sin_cos_known_angles():
    """0 degrees is (0, 1) and 90 degrees is (1, 0)."""
    assert _deg_to_sin_cos(0) == pytest.approx((0, 1))
    assert _deg_to_sin_cos(90) == pytest.approx((1, 0))


def test_deg_to_sin_cos_continuity():
    """Test that the function is continuous at 360 degrees.359 to 1 degree"""
    assert _deg_to_sin_cos(359) == pytest.approx(_deg_to_sin_cos(-1))
    assert _deg_to_sin_cos(360) == pytest.approx(_deg_to_sin_cos(0))


def test_representation_and_string_methods():
    """Test the __repr__ and __str__ methods of WindSite"""
    site = WindSite(1)
    assert repr(site) == f"WindSite(site_index=1, path='{site.path}')"
    assert str(site) == f"WindSite 1 with data from {site.path}"


def test_plot_variable_returns_figure_and_axes():
    """Test that plot_variable returns a matplotlib figure and axes"""
    site = WindSite(1)
    fig, ax = site.plot_variable("windspeed_10m", "2020-01-01", "2020-01-02")
    assert len(ax.lines[0].get_xdata()) == 48
    assert ax.get_ylabel() == "windspeed_10m[m/s]"
    plt.close(fig)  # Close the figure after the test to free memory


def test_plot_variable_invalid_range_raises():
    """Test that plot_variable raises ValueError for an invalid date range"""
    site = WindSite(1)
    with pytest.raises(ValueError):
        site.plot_variable("windspeed_10m", "2025-01-01", "2025-01-02")


def test_plot_variable_invalid_variable_raises():
    """Test that plot_variable raises ValueError
    for an invalid variable name"""
    site = WindSite(1)
    with pytest.raises(ValueError):
        site.plot_variable("invalid_variable", "2020-01-01", "2020-01-02")


def test_make_features_shapes_and_columns():
    """Test the shapes and columns of the features and target."""

    site = WindSite(1)
    X, y = site.make_features()
    assert X.shape == (43799, 5)  # 43800 - 1 for the target
    assert y.shape == (43799,)
    assert "winddirection_100m" not in X.columns


def test_make_features_target_is_next_hour():
    """Test that the target is the power of the next hour."""

    site = WindSite(2)
    X, y = site.make_features()
    assert y.iloc[0] == site.data["Power"].iloc[1]
    assert y.iloc[10] == site.data["Power"].iloc[11]


def test_make_features_horizon():
    """Test the horizon parameter of make_features."""
    site = WindSite(3)
    X, y = site.make_features(horizon=6)
    assert len(X) == 43800 - 6
    assert len(y) == 43800 - 6
    assert y.iloc[0] == site.data["Power"].iloc[6]


def test_split_defaul_sizes():
    """Test the sizes betwen train and and test"""
    site = WindSite(3)
    X_train, X_test, y_train, y_test = site.train_test_split()
    assert len(X_train) == 35040
    assert len(X_test) == 8759


def test_split_no_future_in_train():
    """Training data ends before the test period starts (no leakage)"""
    site = WindSite(3)
    X_train, X_test, _, _ = site.train_test_split("2021-03-01", "2021-03-31")
    assert X_train.index.max() < X_test.index.min()


def test_split_custom_range_sizes():
    """One monthrange give 744 test hours"""
    site = WindSite(3)
    X_train, X_test, y_train, y_test = site.train_test_split(
        "2021-03-01", "2021-03-31"
    )
    assert len(X_test) == 744
    assert len(X_train) == 36456
