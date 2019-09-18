#!/usr/bin/env bash
# Download data
python ../src/download.py single \
--variable orography land_sea_mask soil_type  \
--level_type single \
--output_dir /project/meteo/scratch/S.Rasp/weather-benchmark/raw/constants \
--years 1979 \
--month 01 \
--day 01 \
--time 00:00 \
--custom_fn constants_raw.nc

# Regrid data to 5.625 degree
python ../src/regrid.py \
--input_fns /project/meteo/scratch/S.Rasp/weather-benchmark/raw/constants/constants_raw.nc \
--output_dir /project/meteo/scratch/S.Rasp/weather-benchmark/5.625deg/constants \
--ddeg_out 5.625

# Regrid data to 2.5 degree
python ../src/regrid.py \
--input_fns /project/meteo/scratch/S.Rasp/weather-benchmark/raw/constants/constants_raw.nc \
--output_dir /project/meteo/scratch/S.Rasp/weather-benchmark/2.5deg/constants \
--ddeg_out 2.5