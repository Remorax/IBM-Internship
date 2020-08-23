max_pathlens=(3 4 5 6)

for max_pathlen in "${max_pathlens[@]}";
do
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep.txt" python Attention_twostep.py $max_pathlen Input/data_demarcated.pkl "Output/test_demarcated"$max_pathlen".pkl"
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_wtpath.txt" python Attention_twostep_weightedpath.py $max_pathlen Input/data_demarcated.pkl "Output/test_demarcated_wtpath"$max_pathlen".pkl"
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_uniqpath.txt" python Attention_twostep.py $max_pathlen Input/data_demarcated_uniqpath.pkl "Output/test_demarcated_uniqpath"$max_pathlen".pkl"
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_wtpath_uniqpath.txt" python Attention_twostep_weightedpath.py $max_pathlen Input/data_demarcated_uniqpath.pkl "Output/test_demarcated_uniqpath_wtpath"$max_pathlen".pkl"
done

