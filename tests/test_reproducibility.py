import pytest
import numpy as np
import pandas as pd
import xarray as xr
from datetime import date

# --- Import the reference code from the legacy folder ---
import sys
sys.path.append('tests/data/legacy/')
import mhw as oliver_mhw

# --- Import YOUR code from the src directory ---
from mhw3d import bipolarMhwToolBox as ben_mhw


def generate_synthetic_data():
    """
    Generates a synthetic temperature time series based on the example
    in the ecjoliver/marineHeatWaves README.
    """
    # Create time vector
    t_start = date(1982, 1, 1).toordinal()
    t_end = date(2013, 12, 31).toordinal()
    time = np.arange(t_start, t_end + 1)

    # Create synthetic temperature series
    temp = 15. * np.ones(len(time))
    # Add seasonal cycle
    temp += 5. * np.cos(2 * np.pi * (time - 150) / 365.25)
    # Add trend
    temp += 0.2 * (time - time[0]) / (time[-1] - time[0])
    # Add noise
    temp += 0.5 * np.random.randn(len(time))
    
    # Add three MHWs
    temp[1000:1050] += 4.0  # MHW 1
    temp[4000:4100] += 3.0  # MHW 2
    temp[8000:8050] += 5.0  # MHW 3

    # Convert ordinal time to datetime64 for xarray/pandas
    time_datetime = np.array([date.fromordinal(t) for t in time]).astype('datetime64[ns]')

    return time_datetime, temp


def test_ben_mhw_against_oliver_synthetic():
    """
    This test validates that the ben-MHW toolbox produces results
    identical to the original Oliver et al. code using synthetic data.
    """
    # 1. GENERATE SYNTHETIC DATA
    time, temp = generate_synthetic_data()

    # 2. GET EXPECTED RESULT
    # Run the original Oliver code to get the "ground truth" results.
    expected_events, expected_clim = oliver_mhw.detect(time, temp)
    expected_df = pd.DataFrame(expected_events)

    # 3. PREPARE INPUT FOR THE BEN-MHW TOOLBOX
    # Your toolbox requires 'ssta' and 'severity'. We create these
    # from the reference climatology to isolate the test to event detection.
    ssta = temp - expected_clim['seas']
    severity = ssta / (expected_clim['thresh'] - expected_clim['seas'])

    # Create the xarray Dataset your function needs, with dummy lat/lon coords.
    ds_input = xr.Dataset(
        data_vars={
            'ssta': (('time',), ssta),
            'severity': (('time',), severity)
        },
        coords={'time': time, 'lat': [0], 'lon': [0]}
    ).expand_dims(['lat', 'lon'])

    # 4. RUN YOUR CODE
    ds_actual_events = ben_mhw.calculate_MHWs_metrics(ds_input)

    # 5. FORMAT THE ACTUAL RESULTS FOR COMPARISON
    # Extract the event data and build a clean DataFrame.
    ds_actual_events = ds_actual_events.squeeze(drop=True).dropna('event', how='all')

    actual_df = pd.DataFrame({
        'date_start': ds_actual_events['date_start'].values,
        'date_end': ds_actual_events['date_end'].values,
        'date_peak': ds_actual_events['date_peak'].values,
        'duration': ds_actual_events['duration'].values,
        'intensity_max': ds_actual_events['intensity_max'].values,
        'intensity_mean': ds_actual_events['intensity_mean'].values,
        'intensity_cumulative': ds_actual_events['intensity_cumul'].values,
    })
    
    # 6. ASSERT EQUALITY
    # Select only the columns your code calculates for a fair comparison.
    columns_to_compare = actual_df.columns.tolist()
    expected_df_subset = expected_df[columns_to_compare]
    
    # Ensure data types match to avoid false failures.
    actual_df['duration'] = actual_df['duration'].astype(float)
    expected_df_subset['duration'] = expected_df_subset['duration'].astype(float)

    # Use pandas' testing utility to check for identical results.
    pd.testing.assert_frame_equal(
        expected_df_subset.reset_index(drop=True),
        actual_df.reset_index(drop=True),
        # Add a tolerance for minor floating point differences
        check_exact=False, atol=0.01 
    )
