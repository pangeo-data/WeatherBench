"""
Functions for evaluating forecasts.
"""
import numpy as np
import xarray as xr

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.

    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    return np.sqrt(((error)**2 * weights_lat).mean(mean_dims))

def load_test_data(path, var='z'):
    ds = xr.open_mfdataset(f'{path}/*.nc')[var]
    return ds.sel(time=slice('2017', '2018'))


