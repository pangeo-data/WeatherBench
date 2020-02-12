# Convert to netcdf
for file in /media/rasp/Elements/weather-benchmark/IFS_T21/raw/*; do
  cdo -f nc copy "$file" "${file%.grib}.nc"
done

## Regrid to 5.625 degrees
#python ../src/regrid.py \
#--input_fns /media/rasp/Elements/weather-benchmark/IFS_T42/raw/output_42_pl_2.8125*.nc \
#--output_dir /media/rasp/Elements/weather-benchmark/IFS_T42/ \
#--ddeg_out 5.625