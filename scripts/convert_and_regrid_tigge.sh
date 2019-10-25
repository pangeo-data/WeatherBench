# Convert to netcdf
for file in /data/weather-benchmark/tigge/raw/*; do
  cdo -f nc copy "$file" "${file%.grib}.nc"
done

# Regrid to 5.625 degrees
python ../src/regrid.py \
--input_fns /data/weather-benchmark/tigge/raw/*.nc \
--output_dir /data/weather-benchmark/tigge/5.625deg \
--ddeg_out 5.625