from ecmwfapi import ECMWFDataServer
import calendar
server = ECMWFDataServer()
years = [2018]#[2017, 2018]
months = range(1, 13)
for year in years:
    for month in months:
        days = calendar.monthrange(year, month)[1]
        month = str(month).zfill(2)
        server.retrieve({
            "class": "ti",
            "dataset": "tigge",
            "date": f"{year}-{month}-01/to/{year}-{month}-{days}",
            "expver": "prod",
            "grid": "0.5/0.5",
            "levelist": "500",
            "levtype": "pl",
            "origin": "ecmf",
            "param": "156",
            "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120",
            "time": "00:00:00/12:00:00",
            "type": "cf",
            "target": f"/data/weather-benchmark/tigge/z500_{year}_{month}_0.5deg.grib",
        })