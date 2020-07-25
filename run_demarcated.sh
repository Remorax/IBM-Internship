#!/bin/bash

# parents=(7)
# children=(8 10 15 18 21)
# obj_neighbours=(5 8 11)
# dtype_neighbours=(2 4 8)
parents=(7)
children=(21)
obj_neighbours=(11)
dtype_neighbours=(8)


for a in "${parents[@]}";
do
	for b in "${children[@]}";
	do
		for c in "${obj_neighbours[@]}";
		do	
		    for d in "${dtype_neighbours[@]}";
		    do
			    jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_demarcated.txt" python Attention_demarcated.py $a $b $c $d data_demarcated.pkl "test_demarcated.pkl"
			    jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_demarcated_softmax.txt" python Attention_demarcated_softmax.py $a $b $c $d data_demarcated.pkl "test_demarcated_softmax.pkl"
			    jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_demarcated_sumnormalized.txt" python Attention_demarcated_sumnormalized.py $a $b $c $d data_demarcated.pkl "test_demarcated_sumnormalized.pkl"
			    jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_demarcated_unv.txt" python Attention_demarcated_unv.py $a $b $c $d data_demarcated.pkl "test_demarcated_unv.pkl"
			done
		done
	done
done
