import cdsapi
import argparse
import os

all_years = [
    '1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993',
    '1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008',
    '2009','2010','2011','2012','2013','2014','2015','2016','2017','2018'
]
all_months = [
    '01','02','03','04','05','06','07','08','09','10','11','12'
]
all_days = [
    '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
    '16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'
]
all_times = [
    '00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00',
    '09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00',
    '18:00','19:00','20:00','21:00','22:00','23:00'
]
def download_single_file(
        variable,
        level_type,
        output_dir,
        year,
        pressure_level=[],
        month=all_months,
        day=all_days,
        time=all_times,
        custom_fn=None
):
    """
    Download a single file from the ERA5 archive.

    :param variable: Name of variable in archive
    :param level_type: 'single' or 'pressure'
    :param output_dir: Directory where file is stored
    :param year: Year(s) to download data
    :param pressure_level: Pressure levels to download. None for 'single' output type.
    :param month: Month(s) to download data
    :param day: Day(s) to download data
    :param time: Hour(s) to download data. Format: 'hh:mm'
    :param custom_fn: If not None, use custom file name. Otherwise infer from parameters.
    """

    fn = custom_fn or (
        '_'.join(variable + pressure_level + year) + '_raw.nc'
    )

    c = cdsapi.Client()

    request_parameters = {
        'product_type':   'reanalysis',
        'format':         'netcdf',
        'variable':       variable,
        'year':           year,
        'month':          month,
        'day':            day,
        'time':           time,
        'grid':           [0.703125, 0.703125],
    }
    request_parameters.update({'pressure_level': pressure_level} if level_type == 'pressure' else {})

    c.retrieve(
        f'reanalysis-era5-{level_type}-levels',
        request_parameters,
        output_dir + '/' + fn
    )

    print(f"Saved file: {output_dir + '/' + fn}")


def download_years_separately(
        variable,
        level_type,
        output_dir,
        years,
        pressure_level=[],
        month=all_months,
        day=all_days,
        time=all_times,
        custom_fn = None
):
    """
    Download several files from the ERA5 archive. Loops over list of years.

    :param variable: Name of variable in archive
    :param level_type: 'single' or 'pressure'
    :param output_dir: Directory where file is stored
    :param years: Years to download data. Each year is saved separately
    :param pressure_level: Pressure levels to download. None for 'single' output type.
    :param month: Month(s) to download data
    :param day: Day(s) to download data
    :param time: Hour(s) to download data. Format: 'hh:mm'
    :param custom_fn: If not None, use custom file name. Otherwise infer from parameters.
    """
    for year in years:
        download_single_file(
            variable=variable,
            level_type=level_type,
            output_dir=output_dir,
            year=[year],
            pressure_level=pressure_level,
            month=month,
            day=day,
            time=time,
            custom_fn=custom_fn if custom_fn is None else custom_fn.rstrip('.nc') + year + '.nc'
        )

def main(
        mode,
        variable='geopotential',
        level_type='pressure',
        output_dir='./',
        years='1979',
        pressure_level=[],
        month=all_months,
        day=all_days,
        time=all_times,
        custom_fn = None
):
    """
    Command line script to download single or several files from the ERA5 archive.

    :param mode: 'single' or 'several'. If 'several', loops over years
    :param variable: Name of variable in archive
    :param level_type: 'single' or 'pressure'
    :param output_dir: Directory where file is stored
    :param years: Years to download data. Each year is saved separately
    :param pressure_level: Pressure levels to download. None for 'single' output type.
    :param month: Month(s) to download data
    :param day: Day(s) to download data
    :param time: Hour(s) to download data. Format: 'hh:mm'
    :param custom_fn: If not None, use custom file name. Otherwise infer from parameters.
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if level_type == 'pressure':
        assert pressure_level is not None, 'Pressure level must be defined.'

    if mode == 'single':
        download_single_file(
            variable=variable,
            level_type=level_type,
            output_dir=output_dir,
            year=years,
            pressure_level=pressure_level,
            month=month,
            day=day,
            time=time,
            custom_fn=custom_fn
        )
    if mode == 'separate':
        download_years_separately(
            variable=variable,
            level_type=level_type,
            output_dir=output_dir,
            years=years,
            pressure_level=pressure_level,
            month=month,
            day=day,
            time=time,
            custom_fn=custom_fn
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        type=str,
        help="'single' or 'separate'. If 'several', loops over years.",
    )
    parser.add_argument(
        '--variable',
        type=str,
        nargs='+',
        help='Name of variable(s) in archive',
        required=True
    )
    parser.add_argument(
        '--level_type',
        type=str,
        help="'single' or 'pressure'",
        required=True
    )
    parser.add_argument(
        '--pressure_level',
        type=str,
        nargs='+',
        help="Pressure levels to download. None for 'single' output type.",
        default=[]
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help="Directory where file is stored",
        required=True
    )
    parser.add_argument(
        '--years',
        type=str,
        nargs='+',
        help='Years to download data. Each year is saved separately if mode is "separate". Name of variable(s) in archive',
        default=all_years
    )
    parser.add_argument(
        '--month',
        type=str,
        nargs='+',
        help='Month(s) to download data',
        default=all_months
    )
    parser.add_argument(
        '--day',
        type=str,
        nargs='+',
        help='Day(s) to download data',
        default=all_days
    )
    parser.add_argument(
        '--time',
        type=str,
        nargs='+',
        help='Time(s) to download data',
        default=all_times
    )
    parser.add_argument(
        '--custom_fn',
        type=str,
        help='If not None, use custom file name. Otherwise infer from parameters.',
        default=None
    )
    args = parser.parse_args()

    main(
        mode=args.mode,
        variable=args.variable,
        level_type=args.level_type,
        output_dir=args.output_dir,
        years=args.years,
        pressure_level=args.pressure_level or [],
        month=args.month,
        day=args.day,
        time=args.time,
        custom_fn=args.custom_fn
    )

