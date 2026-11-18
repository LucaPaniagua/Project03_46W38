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
