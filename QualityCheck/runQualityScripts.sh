#!/bin/bash

STARTM=`date -u "+%s"`


year=2023

month_first=9
month_last=10

day_first=20
day_last=26

twin1=20
twin2=1

bin=run_new

path="/scratch/brotzer/${bin}/"

#BAR='##################################################'

for cha in BJZ BJU BJV BJW; do
# for cha in BJW; do

	echo -ne "\r${BAR:0:$cha}" 
    
    for month in $(seq $month_first $month_last); do
        
        ## extend month to two digits
        if [ "$month" -lt 10 ]; then
            month="0${month}"
        fi
        
        ## check if required paths exist, otherwise create it
        if [[ ! -d "${path}Qfiles/${year}-${month}" ]]; then
            echo -e "\n creating folder: Qfiles/${year}-${month} \n"
            mkdir "${path}Qfiles/${year}-${month}"
        fi    
        if [[ ! -d "${path}Cfiles/${year}-${month}" ]]; then
            echo -e "\n creating folder: Cfiles/${year}-${month} \n"
            mkdir "${path}Cfiles/${year}-${month}"
        fi
        if [[ ! -d "${path}QHeli/${year}-${month}" ]]; then
            echo -e "\n creating folder: QHeli/${year}-${month} \n"
            mkdir "${path}QHeli/${year}-${month}"
        fi
        if [[ ! -d "${path}QPlots/${year}-${month}" ]]; then
            echo -e "\n creating folder: QPlots/${year}-${month} \n"
            mkdir "${path}QPlots/${year}-${month}"
        fi 
        if [[ ! -d "${path}QPlots/${year}-${month}/${cha}" ]]; then
            echo -e "\n creating folder: QPlots/${year}-${month}/${cha} \n"
            mkdir "${path}QPlots/${year}-${month}/${cha}"
        fi        


        ## loop over days in month and call python scripts
        for day in $(seq $day_first $day_last); do 


            ## extend day to two digits
            if [ "$day" -lt 10 ]; then
                day="0${day}"
            fi
            
  
            echo -e "\n ________________________________________________"
            echo -e "\n Processing ${year}-${month}-${day} ${cha}... \n"
            
            if [[ ! -f "${path}Qfiles/${year}-${month}/${year}-${month}-${day}.Q${cha:2:3}" ]]; then

                echo -e "\n --> creating Qfiles ...\n"
                python3 romy_CreateQualityData.py $year-$month-$day $cha $path $twin1 $twin2
            else
                echo -e "\n ${path}Qfiles/${year}-${month}/${year}-${month}-${day}.Q${cha:2:3} already exists!\n"
            fi
            
            ## check if .missing exists, which 'romy_CreateQualityData.py' creates if no data for year.doy 
            if [[ ! -f "${path}${year}${month}${day}.missing.txt" ]]; then

                echo -e "\n --> evaluating Qfiles ...\n"
                python3 romy_EvaluateQualityData.py $year-$month-$day $cha $path

                echo -e "\n --> Creating helicorder plots ...\n"
                python3 romy_QualityHelicorder.py $year-$month-$day $cha $path
            
            else
                if [[ ! -f "${path}${year}-${cha}.log" ]]; then 
                    touch "${path}${year}-${cha}.log"
                fi
                
                echo "${year}-${month}-${day}, ${cha}, missingdata" >> ${path}${year}-${cha}.log
               
                rm "${path}${year}${month}${day}.missing.txt"
                   
            fi

        done
    done
done


STOPM=`date -u "+%s"`

RUNTIMEM=`expr $STOPM - $STARTM`

if (($RUNTIMEM>59)); then
    TTIMEM=`printf "%dm%ds\n" $((RUNTIMEM/60%60)) $((RUNTIMEM%60))`
else
    TTIMEM=`printf "%ds\n" $((RUNTIMEM))`
fi

echo -e "\nExecuting "script function" took: $TTIMEM \n"

## End of File
