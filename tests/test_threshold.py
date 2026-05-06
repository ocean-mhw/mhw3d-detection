"""
Regression tests for legacy.compute_climatology and legacy.compute_threshold
against Oliver's reference implementation (marineHeatWaves.detect).

Uses synthetic data from leap years only (1984, 1988, ..., 2012).  Every date
in the dataset falls in a real leap year, so Python's dt.dayofyear and Oliver's
leap-year-2012 DOY convention agree for all 366 days — no off-by-one after Feb 28.

Oliver's event-detection step is also called (it runs after climatology internally)
but its output is not tested here; only clim['seas'] and clim['thresh'] are used.
"""

import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.append("tests/data/legacy/")
import marineHeatWaves as oliver_mhw

from mhw3d import legacy


def _generate_leap_year_data(seed=42):
    """
    Synthetic daily temperature spanning 8 complete leap years (1984–2012),
    with 3-year gaps between them.  All 366 DOYs present; no non-leap-year dates.
    """
    np.random.seed(seed)
    leap_years = list(range(1984, 2013, 4))
    dates = []
    for yr in leap_years:
        dates.extend(pd.date_range(f"{yr}-01-01", f"{yr}-12-31", freq="D"))

    time_dt = np.array(dates).astype("datetime64[ns]")
    t_ord = np.array([pd.Timestamp(d).to_pydatetime().toordinal() for d in time_dt])

    temp = 15.0 * np.ones(len(time_dt))
    temp += 5.0 * np.cos(2 * np.pi * (t_ord - t_ord[0] - 150) / 365.25)
    temp += 0.5 * np.random.randn(len(time_dt))
    return time_dt, t_ord, temp


def _oliver_per_doy(clim_arr, t_ord):
    """
    Oliver's detect() returns time-expanded arrays (length T).
    Map back to a 366-element DOY array using Oliver's own leap-year-2012 mapping.
    Takes the first occurrence of each DOY (all occurrences are identical after smoothing).
    """
    yr_lp = 2012
    t_lp = np.arange(date(yr_lp, 1, 1).toordinal(), date(yr_lp, 12, 31).toordinal() + 1)
    month_lp = np.array([date.fromordinal(int(t)).month for t in t_lp])
    day_lp   = np.array([date.fromordinal(int(t)).day   for t in t_lp])

    result = np.full(366, np.nan)
    seen: set = set()
    for i, t in enumerate(t_ord):
        d = date.fromordinal(int(t))
        doy = int(np.where((month_lp == d.month) & (day_lp == d.day))[0][0]) + 1
        if doy not in seen:
            result[doy - 1] = clim_arr[i]
            seen.add(doy)
    return result


@pytest.fixture(scope="module")
def leap_year_inputs():
    time_dt, t_ord, temp = _generate_leap_year_data()
    # Pass climatologyPeriod explicitly to avoid Oliver's mutable-default-argument bug:
    # detect() mutates its [None, None] default list in-place, which persists across test runs.
    _, oliver_clim = oliver_mhw.detect(t_ord, temp, climatologyPeriod=[1984, 2012])
    da = xr.DataArray(temp, coords={"time": time_dt}, dims=["time"])
    # Use the full leap-year range as the baseline period for both Oliver and our code.
    return da, t_ord, oliver_clim, (1984, 2012)


def test_climatology_against_oliver(leap_year_inputs):
    da, t_ord, oliver_clim, period = leap_year_inputs
    oliver_seas = _oliver_per_doy(oliver_clim["seas"], t_ord)
    our_seas = legacy.compute_climatology(da, climatologyPeriod=period)
    np.testing.assert_allclose(our_seas.values, oliver_seas, atol=1e-10)


def test_threshold_against_oliver(leap_year_inputs):
    da, t_ord, oliver_clim, period = leap_year_inputs
    oliver_thresh = _oliver_per_doy(oliver_clim["thresh"], t_ord)
    our_thresh = legacy.compute_threshold(da, climatologyPeriod=period)
    np.testing.assert_allclose(our_thresh.values, oliver_thresh, atol=1e-10)
