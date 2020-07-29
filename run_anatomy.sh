#!/bin/bash

max_neighbours=(8 10 12 18 21 23 24 28 32 34)
min_neighbours=(1)

for max_limit in "${max_neighbours[@]}";
do
	for min_limit in "${min_neighbours[@]}";
	do
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_anatomy_use.txt" python Attention_anatomy.py $max_limit $min_limit data_anatomy_use.pkl "test_anatomy_use"$max_limit"_"$min_limit".pkl" 300
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_anatomy_bert300.txt" python Attention_anatomy.py $max_limit $min_limit data_anatomy_bert.pkl "test_anatomy_bert300"$max_limit"_"$min_limit".pkl" 300
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_anatomy_bert600.txt" python Attention_anatomy.py $max_limit $min_limit data_anatomy_bert.pkl "test_anatomy_bert600"$max_limit"_"$min_limit".pkl" 600
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_anatomy_sent2vec300.txt" python Attention_anatomy.py $max_limit $min_limit data_anatomy_sent2vec.pkl "test_anatomy_sent2vec300"$max_limit"_"$min_limit".pkl" 300
	        jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Output_att"$max_limit"_"$min_limit"_anatomy_sent2vec600.txt" python Attention_anatomy.py $max_limit $min_limit data_anatomy_sent2vec.pkl "test_anatomy_sent2vec600"$max_limit"_"$min_limit".pkl" 600
	done
done
