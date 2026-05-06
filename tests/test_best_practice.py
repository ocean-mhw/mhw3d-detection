"""
Tests for mhw3d.best_practice — detrending and baseline selection.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mhw3d import best_practice


def _make_da(slope=0.02, seed=42, start="1982", end="2013"):
    """Synthetic daily SST with a known linear trend and seasonal cycle."""
    np.random.seed(seed)
    time = pd.date_range(start, end, freq="D", inclusive="left")
    n = len(time)
    t = np.arange(n, dtype=float)
    data = (
        15.0
        + 5.0 * np.cos(2 * np.pi * t / 365.25)   # seasonal
        + slope * t                                 # known linear trend
        + 0.3 * np.random.randn(n)                 # noise
    )
    return xr.DataArray(data, coords={"time": time}, dims=["time"], name="sst")


class TestDetrend:
    def test_removes_linear_trend(self):
        slope = 0.02
        da = _make_da(slope=slope)
        detrended = best_practice.detrend(da, period=slice("1982", "2011"))

        # After detrending, a linear fit over the full record should have
        # a slope very close to zero.
        t = np.arange(len(detrended), dtype=float)
        fit = np.polyfit(t, detrended.values, deg=1)
        assert abs(fit[0]) < 1e-6, f"Residual slope too large: {fit[0]:.2e}"

    def test_zero_slope_data_near_zero_mean(self):
        da = _make_da(slope=0.0)
        detrended = best_practice.detrend(da, period=slice("1982", "2011"))
        # detrend subtracts the full fitted line (slope + intercept), so
        # the result is centred near zero. The residual linear slope should
        # still be negligible.
        t = np.arange(len(detrended), dtype=float)
        fit = np.polyfit(t, detrended.values, deg=1)
        assert abs(fit[0]) < 1e-6, f"Residual slope too large: {fit[0]:.2e}"
        assert abs(detrended.mean().item()) < 0.1

    def test_preserves_shape_and_coords(self):
        da = _make_da()
        detrended = best_practice.detrend(da, period=slice("1982", "2011"))
        assert detrended.dims == da.dims
        assert detrended.shape == da.shape
        np.testing.assert_array_equal(detrended.time.values, da.time.values)

    def test_period_none_uses_full_record(self):
        da = _make_da(slope=0.02)
        # Should not raise; result shape must match input.
        detrended = best_practice.detrend(da, period=None)
        assert detrended.shape == da.shape

    def test_baseline_period_does_not_restrict_output(self):
        da = _make_da()
        detrended = best_practice.detrend(da, period=slice("1982", "2011"))
        # Output covers the full record, not just the baseline period.
        assert len(detrended.time) == len(da.time)


class TestBaselinePeriod:
    def test_compute_climatology_baseline_period(self):
        da = _make_da(slope=0.0)
        # Climatology from 1982–1991 vs 2002–2011 should differ when there is
        # a trend, but with slope=0 they should be very close.
        clim_early = best_practice.compute_climatology(da, baseline_period=slice("1982", "1991"))
        clim_late  = best_practice.compute_climatology(da, baseline_period=slice("2002", "2011"))
        assert clim_early.dims == ("dayofyear",)
        assert clim_late.dims  == ("dayofyear",)
        np.testing.assert_allclose(clim_early.values, clim_late.values, atol=0.2)

    def test_compute_threshold_baseline_period(self):
        da = _make_da(slope=0.0)
        thresh = best_practice.compute_threshold(da, baseline_period=slice("1982", "2011"))
        assert "dayofyear" in thresh.dims
        assert thresh.sizes["dayofyear"] == 366

    def test_trending_data_baseline_effect(self):
        # With a strong upward trend, the threshold computed from a late period
        # should be higher than one from an early period.
        da = _make_da(slope=0.05)
        thresh_early = best_practice.compute_threshold(da, baseline_period=slice("1982", "1991"))
        thresh_late  = best_practice.compute_threshold(da, baseline_period=slice("2004", "2013"))
        assert thresh_late.mean() > thresh_early.mean()

    def test_detrend_then_compute_baseline(self):
        # Full workflow: detrend then compute climatology on the baseline period.
        da = _make_da(slope=0.02)
        period = slice("1982", "2011")
        da_dt = best_practice.detrend(da, period=period)
        seas   = best_practice.compute_climatology(da_dt, baseline_period=period)
        thresh = best_practice.compute_threshold(da_dt,   baseline_period=period)
        assert seas.sizes["dayofyear"]   == 366
        assert thresh.sizes["dayofyear"] == 366
        # Threshold should everywhere exceed the climatology.
        assert (thresh > seas).all()
