#!/bin/bash

max_neighbours=(8 10 12 18 21 22 23 32 34)
min_neighbours=(1)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_prop.txt" python Attention_preprocessing.py $max_limit $min_limit data_prop.pkl "test_prop"$max_limit"_"$min_limit".pkl"
	done
done

max_neighbours=(8 10 12 18 21 22 23 32 34 37)
min_neighbours=(1)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_prop_enlarged.txt" python Attention_preprocessing.py $max_limit $min_limit data_prop_enlarged.pkl "test_prop_enlarged"$max_limit"_"$min_limit".pkl"
	done
done

max_neighbours=(8 10 12 18 23 32 36 40 45 50 55)
min_neighbours=(1)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_prop_enlarged_hashed.txt" python Attention_preprocessing.py $max_limit $min_limit data_prop_enlarged_hashed.pkl "test_prop_enlarged_hashed"$max_limit"_"$min_limit".pkl"
	done
done

