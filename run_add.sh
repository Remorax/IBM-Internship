#!/bin/bash

max_neighbours=(8 10 12 18 20 21 22 23 24)
min_neighbours=(2 3)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 16g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_add.txt" python Attention_add.py $max_limit $min_limit
	done
done