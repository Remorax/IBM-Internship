max_pathlens=(12 18 21 22 23 24 26 32 38)
types=(0 1 2 3)

for max_pathlen in "${max_pathlens[@]}";
do
	for type in "${types[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_uncategorized_bagofnbrs.txt" python Attention_twostep_uncategorized.py Input/data_demarcated_bagofnbrs.pkl $max_pathlen $type
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_uncategorized_bagofnbrs_wtpath.txt" python Attention_twostep_uncategorized_weighted.py Input/data_demarcated_bagofnbrs.pkl $max_pathlen $type
	done
done
