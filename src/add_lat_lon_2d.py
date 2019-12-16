import argparse
import xarray as xr
import numpy as np
import xesmf as xe
from glob import glob
import os
import shutil

def add_2d(
        ds,
):
    """
    Regrid horizontally.
    :param ds: Input xarray dataset
    """
    ds['lat2d'] = ds.lat.expand_dims({'lon': ds.lon}).transpose()
    ds['lon2d'] = ds.lon.expand_dims({'lat': ds.lat})
    return ds


def convert_z_to_orography(ds):
    """
    Convert geopotential of surface to height in meters
    Args:
        ds: Input dataset
    Returns:
        ds: Same dataset with orography instead of z
    """
    ds['z'] = ds.z / 9.80665
    ds = ds.rename({'z': 'orography'})
    ds.orography.attrs['units'] = 'm'
    return ds


def main(
        input_fns,
        custom_fn=None,
):
    """
    :param input_fns: Input files. Can use *. If more than one, loop over them
    :param custom_fn: If not None, use custom file name. Otherwise infer from parameters.
    """
    # Get files for starred expressions
    if '*' in input_fns[0]:
        input_fns = sorted(glob(input_fns[0]))
    # Loop over input files
    for fn in input_fns:
        print(f'Extracting from file: {fn}')
        ds = xr.open_dataset(fn).isel(time=0).drop('time')
        ds = convert_z_to_orography(add_2d(ds))
        fn_out = (
            custom_fn or fn
        )
        print(f"Saving file: {fn_out}")
        ds.to_netcdf(fn_out+'.tmp')
        ds.close()
        shutil.move(fn_out+'.tmp', fn_out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_fns',
        type=str,
        nargs='+',
        help="Input files (full path). Can use *. If more than one, loop over them",
        required=True
    )
    parser.add_argument(
        '--custom_fn',
        type=str,
        help="If not None, use custom file name. Otherwise infer from parameters.",
        default=None
    )

    args = parser.parse_args()

    main(
        input_fns=args.input_fns,
        custom_fn=args.custom_fn,
    )



