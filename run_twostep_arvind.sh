max_pathlens=(3 4 5 6)
max_paths=(1 2 3 4 5 6 9 10 12 18 20 21 24 26)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep.txt" python Attention_twostep.py $max_pathlen $max_path Input/data_demarcated.pkl
		jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_wtpath.txt" python Attention_twostep_weightedpath.py $max_pathlen $max_path Input/data_demarcated.pkl
	done
done
