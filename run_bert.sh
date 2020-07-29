#!/bin/bash

max_neighbours=(8 10 12 18 23 32 34)
min_neighbours=(1)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_bert300.txt" python Attention_bert.py $max_limit $min_limit data_bert.pkl "test_bert"$max_limit"_"$min_limit"_300.pkl" 300
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_bert600.txt" python Attention_bert.py $max_limit $min_limit data_bert.pkl "test_bert"$max_limit"_"$min_limit"_600.pkl" 600
	done
done
