#!/bin/bash

max_neighbours=(8 10 12 18 21 22 23 32 36 40 45 48 51)
min_neighbours=(1)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_german_phrase.txt" python Attention_german.py $max_limit $min_limit data_german_dataset_phrase.pkl "test_german_dataset_phrase"$max_limit"_"$min_limit".pkl"
	done
done
