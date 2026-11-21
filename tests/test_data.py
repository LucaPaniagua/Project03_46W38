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
