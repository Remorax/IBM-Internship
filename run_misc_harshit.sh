 # Experimenting with false negative percentages
max_paths=(3 13 21)
max_pathlens=(5 6 8)
thresholds=(0.6075)
false_negatives=(0.55 0.6 0.65 0.7)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		for threshold in "${thresholds[@]}";
		do
			for false_negative in "${false_negatives[@]}";
			do
				# jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$false_negative"_fn_testing.txt" python Attention_fn_testing.py Input/data_conf_oaei_german_aml_thresh.pkl $max_path $max_pathlen $threshold $false_negative
				jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$false_negative"_fn_testing_mini.txt" python Attention_fn_testing_mini.py Input/data_conf_oaei_german_aml_thresh.pkl $max_path $max_pathlen $threshold $false_negative
			done
		done
	done
done
