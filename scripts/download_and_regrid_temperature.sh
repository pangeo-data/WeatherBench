#!/usr/bin/env bash
# Download data
python ../src/download.py separate \
--variable temperature \
--level_type pressure \
--pressure_level 10 100 200 300 400 500 600 700 850 1000 \
--output_dir /media/rasp/Elements/weather-benchmark \
--years 2017

# Regrid data to 5.625 degree
python ../src/regrid.py \
--input_fns /data/weather-benchmark/raw/temperature/* \
--output_dir /data/weather-benchmark/5.625deg/temperature \
--ddeg_out 5.625

# Regrid data to 2.8125 degree
#python ../src/regrid.py \
#--input_fns /data/weather-benchmark/raw/temperature/* \
#--output_dir /data/weather-benchmark/2.8125deg/temperature \
#--ddeg_out 2.8125
