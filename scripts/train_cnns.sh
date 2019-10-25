python ../src/train_nn.py \
/data/weather-benchmark/5.625deg/geopotential_500/ \
/data/weather-benchmark/predictions/fc_cnn_3d.nc \
72 \
[32,64,64,64,64,32,1] \
[5,5,5,5,5,5,5]

#python ../src/train_nn.py \
#/data/weather-benchmark/5.625deg/geopotential_500/ \
#/data/weather-benchmark/predictions/fc_cnn_3d_no_period.nc \
#72 \
#[32,64,64,64,32,1] \
#[5,5,5,5,5,5] \
#--periodic False
#
#python ../src/train_nn.py \
#/data/weather-benchmark/5.625deg/geopotential_500/ \
#/data/weather-benchmark/predictions/fc_cnn_3d_drop.nc \
#72 \
#[32,64,64,64,32,1] \
#[5,5,5,5,5,5] \
#--dropout 0.3

python ../src/train_nn.py \
/data/weather-benchmark/5.625deg/geopotential_500/ \
/data/weather-benchmark/predictions/fc_cnn_5d.nc \
120 \
[32,64,64,64,64,32,1] \
[5,5,5,5,5,5,5]

python ../src/train_nn.py \
/data/weather-benchmark/5.625deg/geopotential_500/ \
/data/weather-benchmark/predictions/fc_cnn_6h_iter.nc \
6 \
[32,64,64,64,64,32,1] \
[5,5,5,5,5,5,5] \
--iterative True