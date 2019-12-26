# Convert to netcdf
for file in /media/rasp/Elements/weather-benchmark/tigge/raw/*.grib; do
  cdo -f nc copy "$file" "${file%.grib}.nc"
done

mkdir /media/rasp/Elements/weather-benchmark/tigge/netcdf
mv /media/rasp/Elements/weather-benchmark/tigge/raw/*.nc /media/rasp/Elements/weather-benchmark/tigge/netcdf/

# Regrid to 5.625 degrees
python ../src/regrid.py \
--input_fns /media/rasp/Elements/weather-benchmark/tigge/netcdf/*.nc \
--output_dir /media/rasp/Elements/weather-benchmark/tigge/5.625deg \
--ddeg_out 5.625

# Regrid to 2.8125 degrees
python ../src/regrid.py \
--input_fns /media/rasp/Elements/weather-benchmark/tigge/netcdf/*.nc \
--output_dir /media/rasp/Elements/weather-benchmark/tigge/2.8125deg \
--ddeg_out 2.8125

# Regrid to 1.40625 degrees
python ../src/regrid.py \
--input_fns /media/rasp/Elements/weather-benchmark/tigge/netcdf/*.nc \
--output_dir /media/rasp/Elements/weather-benchmark/tigge/1.40625deg \
--ddeg_out 1.40625