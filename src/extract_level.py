import argparse
import xarray as xr
import numpy as np
import xesmf as xe
from glob import glob
import os

def extract(
        ds_in,
        level
):
    """
    Regrid horizontally.
    :param ds_in: Input xarray dataset
    :param level: level to extract
    """
    ds_out = ds_in.sel(level=level)
    return ds_out


def main(
        input_fns,
        output_dir,
        level,
        custom_fn=None,
):
    """
    :param input_fns: Input files. Can use *. If more than one, loop over them
    :param output_dir: Output directory
    :param level: pressure level to extract
    :param custom_fn: If not None, use custom file name. Otherwise infer from parameters.
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Get files for starred expressions
    if '*' in input_fns[0]:
        input_fns = sorted(glob(input_fns[0]))
    # Loop over input files
    for fn in input_fns:
        print(f'Extracting from file: {fn}')
        ds_in = xr.open_dataset(fn)
        ds_out = extract(ds_in, level)
        # Assume name like "geopotential_1979_5.625deg.nc"
        fn_parts = fn.split('/')[-1].split('_')
        fn_out = (
            custom_fn or
            '_'.join([fn_parts[0], f'{level}hPa'] + fn_parts[1:])
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
        '--level',
        type=int,
        help="Pressure level to extract",
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
        output_dir=args.output_dir,
        level=args.level,
        custom_fn=args.custom_fn,
    )



