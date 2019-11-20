# Convert to netcdf
for file in /media/rasp/Elements/weather-benchmark/IFS_T21/raw/*; do
  cdo -f nc copy "$file" "${file%.grib}.nc"
done

# Regrid to 5.625 degrees
python ../src/regrid.py \
--input_fns /media/rasp/Elements/weather-benchmark/IFS_T21/raw/*.nc \
--output_dir /media/rasp/Elements/weather-benchmark/IFS_T21/5.625deg \
--ddeg_out 5.625