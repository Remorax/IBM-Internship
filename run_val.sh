#!/bin/bash

max_neighbours=(12 18 21 22 23 24 26 32 38)
min_neighbours=(1)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_limit"_"$min_limit"_onestep.txt" python Attention_val.py $max_limit $min_limit Input/data_multi_rootpath.pkl
	done
done
