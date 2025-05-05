#! /usr/bin/bash
cd /hpi/fs00/home/zi.yang/research_project/ArchesWeather
#conda activate weather
for((i=2002;i<=2007;i++));
do
python dl_era.py --year $i --folder /hpi/fs00/share/ekapex/era5_240
done
