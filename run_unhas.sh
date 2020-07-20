#!/bin/bash

max_neighbours=(8 10 12 18 21 22 23)
min_neighbours=(2 3)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_unhas.txt" python Attention_opt_unhas.py $max_limit $min_limit data_unhas.pkl
            jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_spellchecked.txt" python Attention_opt_unhas.py $max_limit $min_limit data_v2.pkl
	done
done
