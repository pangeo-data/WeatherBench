#!/usr/bin/env bash
# Download data
#python ../src/download.py separate \
#--variable geopotential \
#--level_type pressure \
#--pressure_level 500 \
#--output_dir /project/meteo/scratch/S.Rasp/weather-benchmark/raw/geopotential_500 \
#--years 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018
#
# Regrid data to 5.625 degree
python ../src/regrid.py \
--input_fns /project/meteo/scratch/S.Rasp/weather-benchmark/raw/geopotential_500/* \
--output_dir /project/meteo/scratch/S.Rasp/weather-benchmark/5.625deg/geopotential_500 \
--ddeg_out 5.625

# Regrid data to 2.8125 degree
python ../src/regrid.py \
--input_fns /project/meteo/scratch/S.Rasp/weather-benchmark/raw/geopotential_500/* \
--output_dir /project/meteo/scratch/S.Rasp/weather-benchmark/2.8125deg/geopotential_500 \
--ddeg_out 2.8125
