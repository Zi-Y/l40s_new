#! /usr/bin/bash
cd /hpi/fs00/home/zi.yang/research_project/ArchesWeather
#conda activate weather
for((i=1979;i<=2000;i++));
do
python dl_era.py --year $i --folder /hpi/fs00/share/ekapex/era5_240
done
