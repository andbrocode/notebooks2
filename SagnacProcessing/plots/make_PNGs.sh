#!/bin/bash

for i in $(ls *.pdf); do 

echo $i 

if [ -f "png/${i%.*}.png" ]; then 
	echo "skipped"
	continue
else 
	pdf2png $i ${i%.*}.png 
fi

done

mv *.png png/
