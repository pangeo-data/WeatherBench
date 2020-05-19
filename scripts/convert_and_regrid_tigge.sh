echo "Dir = $1"
echo "Var = $2"

# Regrid to 5.625 degrees
python ../src/regrid.py \
--input_fns "$1"/raw/"$2"/*.grib \
--output_dir "$1"/5.625deg/"$2" \
--ddeg_out 5.625 \
--is_grib 1



## Convert to netcdf
#for file in "$1"/raw/"$2"/*.grib; do
#  cdo -f nc copy "$file" "${file%.grib}.nc"
#done
#
#mkdir "$1"/netcdf
#mv "$1"/raw/"$2"/*.nc "$1"/netcdf/"$2"/
#
## Regrid to 5.625 degrees
#python ../src/regrid.py \
#--input_fns "$1"/netcdf/"$2"/*.nc \
#--output_dir "$1"/5.625deg/"$2" \
#--ddeg_out 5.625
#
## Regrid to 2.8125 degrees
#python ../src/regrid.py \
#--input_fns "$1"/netcdf/"$2"/*.nc \
#--output_dir "$1"/2.8125deg/"$2" \
#--ddeg_out 2.8125
#
## Regrid to 1.40625 degrees
#python ../src/regrid.py \
#--input_fns "$1"/netcdf/"$2"/*.nc \
#--output_dir "$1"/1.40625deg/"$2" \
#--ddeg_out 1.40625
