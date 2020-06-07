import argparse
import xarray as xr
import numpy as np
import xesmf as xe
from glob import glob
import os
from tqdm import tqdm

def regrid(
        ds_in,
        ddeg_out,
        method='bilinear',
        reuse_weights=True
):
    """
    Regrid horizontally.
    :param ds_in: Input xarray dataset
    :param ddeg_out: Output resolution
    :param method: Regridding method
    :param reuse_weights: Reuse weights for regridding
    :return: ds_out: Regridded dataset
    """
    # Rename to ESMF compatible coordinates
    if 'latitude' in ds_in.coords:
        ds_in = ds_in.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Create output grid
    grid_out = xr.Dataset(
        {
            'lat': (['lat'], np.arange(-90+ddeg_out/2, 90, ddeg_out)),
            'lon': (['lon'], np.arange(0, 360, ddeg_out)),
        }
    )

    # Create regridder
    regridder = xe.Regridder(
        ds_in, grid_out, method, periodic=True, reuse_weights=reuse_weights,
    )

    # Hack to speed up regridding of large files
    ds_list = []
    chunk_size = 500
    n_chunks = len(ds_in.time) // chunk_size + 1
    for i in range(n_chunks):
        ds_small = ds_in.isel(time=slice(i*chunk_size, (i+1)*chunk_size))
        ds_list.append(regridder(ds_small).astype('float32'))
    ds_out = xr.concat(ds_list, dim='time')

    # Set attributes since they get lost during regridding
    for var in ds_out:
        ds_out[var].attrs =  ds_in[var].attrs
    ds_out.attrs.update(ds_in.attrs)

    # # Regrid dataset
    # ds_out = regridder(ds_in)
    return ds_out


def main(
        input_fns,
        output_dir,
        ddeg_out,
        method='bilinear',
        reuse_weights=True,
        custom_fn=None,
        file_ending='nc',
        is_grib=False
):
    """
    :param input_fns: Input files. Can use *. If more than one, loop over them
    :param output_dir: Output directory
    :param ddeg_out: Output resolution
    :param method: Regridding method
    :param reuse_weights: Reuse weights for regridding
    :param custom_fn: If not None, use custom file name. Otherwise infer from parameters.
    :param file_ending: Default = nc
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Get files for starred expressions
    if '*' in input_fns[0]:
        input_fns = sorted(glob(input_fns[0]))
    # Loop over input files
    for fn in tqdm(input_fns):
        print(f'Regridding file: {fn}')
        if is_grib:
            ds_in = xr.open_dataset(fn, engine='cfgrib')
        else:
            ds_in = xr.open_dataset(fn)
        ds_out = regrid(ds_in, ddeg_out, method, reuse_weights)
        fn_out = (
            custom_fn or
            '_'.join(fn.split('/')[-1][:-3].split('_')[:-1]) + '_' + str(ddeg_out) + 'deg.' + file_ending
        )
        print(f"Saving file: {output_dir + '/' + fn_out}")
        ds_out.to_netcdf(output_dir + '/' + fn_out)
        ds_in.close(); ds_out.close()

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
        '--output_dir',
        type=str,
        help="Output directory",
        required=True
    )
    parser.add_argument(
        '--ddeg_out',
        type=float,
        help="Output resolution",
        required=True
    )
    parser.add_argument(
        '--reuse_weights',
        type=int,
        help="Reuse weights for regridding. 0 or 1 (default)",
        default=1  
    )
    parser.add_argument(
        '--custom_fn',
        type=str,
        help="If not None, use custom file name. Otherwise infer from parameters.",
        default=None
    )
    parser.add_argument(
        '--file_ending',
        type=str,
        help="File ending. Default = nc",
        default='nc'
    )
    parser.add_argument(
        '--is_grib',
        type=int,
        help="Input is .grib file. 0 (default) or 1",
        default=0
    )
    args = parser.parse_args()

    main(
        input_fns=args.input_fns,
        output_dir=args.output_dir,
        ddeg_out=args.ddeg_out,
        reuse_weights=args.reuse_weights,
        custom_fn=args.custom_fn,
        file_ending=args.file_ending,
        is_grib=args.is_grib
    )



