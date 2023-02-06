#!/bin/bash

year=2019
#month=11
#cha=BJZ

#BAR='##################################################'

#for cha in BJZ BJU BJV BJW; do
for cha in BJZ; do
#	echo -ne "\r${BAR:0:$cha}" 
    for month in {1..12}; do
	echo $month
#	for day in {1..31}; do 
	for day in {1..5}; do 

		echo -e "\n ________________________________________________"

		echo -e "\n Processing ${year}-${month}-${day}... \n"
		python3 romy_CreateQualityData.py $year-$month-$day $cha

		echo -e "\n Evaluating ...\n"
		python3 romy_EvaluateQualityData.py $year-$month-$day $cha

		echo -e "\n Creating helicorder plots ...\n"
		python3 romy_QualityHelicorder.py $year-$month-$day $cha


	done
    done
done



## End of File
