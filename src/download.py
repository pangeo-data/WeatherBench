import cdsapi
import fire


def download_single_file(
        variable,
        level_type,
        output_dir,
        year,
        pressure_level=None,
        month=[
            '01','02','03','04','05','06','07','08','09','10','11','12'
        ],
        day=[
            '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
            '16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'
        ],
        time=[
            '00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00',
            '09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00',
            '18:00','19:00','20:00','21:00','22:00','23:00'
        ],
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

    # TODO: This is an ugly contraption. Should get better with a proper argparse
    fn = custom_fn or (
        '_'.join(list(map(str,
            (variable if type(variable) is list else [variable]) +
            (pressure_level if type(pressure_level) is list else [pressure_level]) +
            (year if type(year) is list else [year])
        ))) + '.nc'
    )

    c = cdsapi.Client()

    c.retrieve(
        f'reanalysis-era5-{level_type}-levels',
        {
            'product_type':   'reanalysis',
            'format':         'netcdf',
            'pressure_level': pressure_level,
            'variable':       variable,
            'year':           year,
            'month':          month,
            'day':            day,
            'time':           time
        },
        output_dir + '/' + fn
    )
    print(f"Saved file: {output_dir + '/' + fn}")


def download_years_separately(
        variable,
        level_type,
        output_dir,
        years,
        pressure_level=None,
        month=[
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'
        ],
        day=[
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
            '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'
        ],
        time=[
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ],
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
            year=year,
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
        pressure_level=None,
        month=[
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'
        ],
        day=[
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
            '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'
        ],
        time=[
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ],
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
    # TODO: Fix this by using argparse
    # years = years if type(years) is list else [years]
    # years = [str(year) for year in years]

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
    if mode == 'several':
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
    fire.Fire(main)