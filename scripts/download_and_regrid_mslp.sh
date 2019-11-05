#!/usr/bin/env bash
# Download data
python ../src/download.py separate \
--variable mean_sea_level_pressure \
--level_type single \
--output_dir /data/weather-benchmark/raw/mean_sea_level_pressure \
--years 2017

# Regrid data to 5.625 degree
python ../src/regrid.py \
--input_fns /data/weather-benchmark/raw/mean_sea_level_pressure/* \
--output_dir /data/weather-benchmark/5.625deg/mean_sea_level_pressure \
--ddeg_out 5.625

# Regrid data to 2.8125 degree
#python ../src/regrid.py \
#--input_fns /data/weather-benchmark/raw/mean_sea_level_pressure/* \
#--output_dir /data/weather-benchmark/2.8125deg/mean_sea_level_pressure \
#--ddeg_out 2.8125
