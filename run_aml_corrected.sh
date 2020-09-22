# # Train on AML conference + German dataset v2

max_paths=(3 13 21 26)
max_pathlens=(5 8)
thresholds=(0.6075 0.612)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		for threshold in "${thresholds[@]}";
		do
			jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$threshold"_aml_corrected.txt" python Attention_aml_corrected.py Input/data_conf_oaei_german_aml_corrected.pkl $max_path $max_pathlen $threshold  "Output/test_aml_corrected_"$max_path"_"$max_pathlen"_"$threshold".pkl" "Models/aml_corrected_"$max_path"_"$max_pathlen"_"$threshold".pt"
		done
	done
done