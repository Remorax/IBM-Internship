#!/bin/bash

max_neighbours=(8 10 12 15 18 24 30 36 50 72 100 112 128 144 150 156)
min_neighbours=(1 2)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_generic_rootpath_.txt" python Attention_generic_rootpath.py $max_limit $min_limit data_generic_rootpath.pkl "test_generic_rootpath"$max_limit"_"$min_limit".pkl"
	done
done
