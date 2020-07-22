#!/bin/bash

max_neighbours=(2 3 4 5 6 7)
min_neighbours=(1 2 3 4)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_path.txt" python Attention_path.py $max_limit $min_limit data_path.pkl "test_path"$max_limit"_"$min_limit".pkl"
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_path_no_repeat.txt" python Attention_path_no_repeat.py $max_limit $min_limit data_path.pkl "test_path_no_repeat"$max_limit"_"$min_limit".pkl"
	done
done
