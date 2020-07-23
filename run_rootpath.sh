#!/bin/bash

max_neighbours=(8 10 12 18 21 22 23 24 26 28 30 32 34)
min_neighbours=(1)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_rootpath_enlarged.txt" python Attention_preprocessing.py $max_limit $min_limit data_enlarged_rootpath.pkl "test_rootpath"$max_limit"_"$min_limit".pkl"
	done
done
