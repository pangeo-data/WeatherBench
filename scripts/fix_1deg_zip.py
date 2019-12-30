from glob import glob
import os
BASEDIR = '/home/rasp/mediaTUM/'
res = '1.40625'
subdirs = glob(f'{BASEDIR}{res}deg/*/')
subdirs = [sd for sd in subdirs if not 'constants' in sd]
for sd in subdirs:
    var = sd.split('/')[-2]
    os.chdir(sd)
    print(os.listdir())
    zipname = f'{var}_{res}deg.zip'
    os.system(f'mv {zipname} old_{zipname}')
    os.system(f'unzip old_{zipname}')
    os.system(f'mv media/rasp/Elements/weather-benchmark/1.40625deg/{var}/*.nc ./')
    os.system(f'rmdir -p media/rasp/Elements/weather-benchmark/1.40625deg/{var}/')
    os.system(f'zip {zipname} *.nc')
    if var != 'geopotential':
        os.system(f'rm ./*.nc')