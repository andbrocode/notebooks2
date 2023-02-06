#!/bin/bash

path='/home/brotzer/Documents/ROMY/ROMY_QualityCheck/stations/'

#year="2019"

#sta="ROMY"

# TON
for sta in ALFT GELB BIB; do 

for year in 2018 2019 2020 ; do 

#	for cha in BJZ BJU BJV BJW ;do
	for cha in BHZ BHN BHE;do
		if [ $cha == "BJZ" ]; then
			code="BW.${sta}.10.${cha}"
		else
			code="BW.${sta}..${cha}"
		fi

		echo -e "\n$year $code ..."

		python3 updateConfig.py "config.ini" "$path" "$code" "$year" 
	
		python3 runTracesQuality.py

		cp config.ini ${path}${year}_${code}_${cha}.config

	done 
done
done
## END OF FILE
