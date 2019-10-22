from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "ti",
    "dataset": "tigge",
    "date": "2016-01-01/to/2016-01-03",
    "expver": "prod",
    "grid": "0.5/0.5",
    "levelist": "500",
    "levtype": "pl",
    "origin": "ecmf",
    "param": "156",
    "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120",
    "time": "00:00:00/12:00:00",
    "type": "cf",
    "target": "/data/weather-benchmark/tigge/z500.nc",
})