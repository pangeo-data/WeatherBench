#!/usr/bin/env bash
# Download data
python ../src/download.py single \
--variable orography land_sea_mask soil_type  \
--level_type single \
--output_dir /media/rasp/Elements/weather-benchmark/raw/constants \
--years 1979 \
--month 01 \
--day 01 \
--time 00:00 \
--custom_fn constants_raw.nc

# Regrid data to 5.625 degree
python ../src/regrid.py \
--input_fns /media/rasp/Elements/weather-benchmark/raw/constants/constants_raw.nc \
--output_dir /media/rasp/Elements/weather-benchmark/5.625deg/constants \
--ddeg_out 5.625

# Regrid data to 2.8125 degree
python ../src/regrid.py \
--input_fns /media/rasp/Elements/weather-benchmark/raw/constants/constants_raw.nc \
--output_dir /media/rasp/Elements/weather-benchmark/2.8125deg/constants \
--ddeg_out 2.8125

# Regrid data to 1.40625 degree
python ../src/regrid.py \
--input_fns /media/rasp/Elements/weather-benchmark/raw/constants/constants_raw.nc \
--output_dir /media/rasp/Elements/weather-benchmark/1.40625deg/constants \
--ddeg_out 1.40625

# Add 2d lat and lon fields
python ../src/add_lat_lon_2d.py \
--input_fns /media/rasp/Elements/weather-benchmark/5.625deg/constants/constants_raw.nc \
--output_dir /media/rasp/Elements/weather-benchmark/5.625deg/constants \

# Regrid data to 2.8125 degree
python ../src/add_lat_lon_2d.py \
--input_fns /media/rasp/Elements/weather-benchmark/2.8125deg/constants/constants_raw.nc \
--output_dir /media/rasp/Elements/weather-benchmark/2.8125deg/constants \

# Regrid data to 1.40625 degree
python ../src/add_lat_lon_2d.py \
--input_fns /media/rasp/Elements/weather-benchmark/1.40625deg/constants/constants_raw.nc \
--output_dir /media/rasp/Elements/weather-benchmark/1.40625deg/constants \

# Final processing
python ../src/add_lat_lon_2d.py \
--input_fns /media/rasp/Elements/weather-benchmark/5.625deg/constants/constants_5.625deg.nc \
/media/rasp/Elements/weather-benchmark/2.8125deg/constants/constants_2.8125deg.nc \
/media/rasp/Elements/weather-benchmark/1.40625deg/constants/constants_1.40625deg.nc