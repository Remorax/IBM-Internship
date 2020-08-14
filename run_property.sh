#!/bin/bash

max_neighbours=(10 12 18 21 22 23 24 28 30 32 34)
min_neighbours=(1)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
        jbsub -q x86_6d -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_prop_concat_sparse.txt" python Attention_prop_concat.py $max_limit $min_limit data_prop_dtpath.pkl "test_prop"$max_limit"_"$min_limit".pkl"
        jbsub -q x86_6d -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_prop_concat.txt" python Attention_prop_concat_sparse.py $max_limit $min_limit data_prop_dtpath.pkl "test_prop_dtpath"$max_limit"_"$min_limit".pkl"
        jbsub -q x86_6d -mem 40g -require v100 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_prop_concat_sparse.txt" python Attention_prop_concat.py $max_limit $min_limit data_prop_dtpath.pkl "test_prop"$max_limit"_"$min_limit".pkl"
        jbsub -q x86_6d -mem 40g -require v100 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_prop_concat.txt" python Attention_prop_concat_sparse.py $max_limit $min_limit data_prop_dtpath.pkl "test_prop_dtpath"$max_limit"_"$min_limit".pkl"
	done
done
