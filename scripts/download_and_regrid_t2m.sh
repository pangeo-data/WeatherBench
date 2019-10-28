#!/usr/bin/env bash
# Download data
python ../src/download.py separate \
--variable 2m_temperature \
--level_type single \
--output_dir /data/weather-benchmark/raw/2m_temperature \
--years 2017

# Regrid data to 5.625 degree
python ../src/regrid.py \
--input_fns /data/weather-benchmark/raw/2m_temperature/* \
--output_dir /data/weather-benchmark/5.625deg/2m_temperature \
--ddeg_out 5.625

# Regrid data to 2.8125 degree
#python ../src/regrid.py \
#--input_fns /data/weather-benchmark/raw/2m_temperature/* \
#--output_dir /data/weather-benchmark/2.8125deg/2m_temperature \
#--ddeg_out 2.8125
