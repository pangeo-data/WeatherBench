import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type':'reanalysis',
        'format':'netcdf',
        'variable':[
            '2m_temperature','land_sea_mask','orography',
            'soil_type','toa_incident_solar_radiation','total_precipitation'
        ],
        'year':[
            '1980','2010'
        ],
        'month':'01',
        'day':'01',
        'time':'00:00'
    },
    'single_test.nc')