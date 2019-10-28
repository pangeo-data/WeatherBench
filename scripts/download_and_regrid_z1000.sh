#!/usr/bin/env bash
# Download data
python ../src/download.py separate \
--variable geopotential \
--level_type pressure \
--pressure_level 1000 \
--output_dir /data/weather-benchmark/raw/geopotential_1000 \
--years 2017

# Regrid data to 5.625 degree
python ../src/regrid.py \
--input_fns /data/weather-benchmark/raw/geopotential_1000/* \
--output_dir /data/weather-benchmark/5.625deg/geopotential_1000 \
--ddeg_out 5.625

# Regrid data to 2.8125 degree
#python ../src/regrid.py \
#--input_fns /data/weather-benchmark/raw/geopotential_1000/* \
#--output_dir /data/weather-benchmark/2.8125deg/geopotential_1000 \
#--ddeg_out 2.8125
