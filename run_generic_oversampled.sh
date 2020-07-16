#!/bin/bash

max_neighbours=(5 8 10 11 12 15 18 20 21 22 23 24)
min_neighbours=(2 3)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_generic_oversampled.txt" python Attention_generic_oversampled.py $max_limit $min_limit
	done
done