# !/bin/bash

# Train on AML conference + German dataset 

max_paths=(7 18)
max_pathlens=(3 6)
thresholds=(0.6 0.6075 0.612 0.627 0.66 0.666 0.694 0.735)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		for threshold in "${thresholds[@]}";
		do
			jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$threshold"_conf_oaei_german_aml.txt" python Attention_german_amlconf_oaei.py Input/data_conf_oaei_german_aml_thresh.pkl $max_path $max_pathlen $threshold  "Output/test_conf_oaei_german_aml_"$max_path"_"$max_pathlen"_"$threshold".pkl" "Models/conf_oaei_german_aml_"$max_path"_"$max_pathlen"_"$threshold".pt"
		done
	done
done

# max_paths=(3 5 7 12 18 21 24 26)
max_paths=(7 18)
# max_pathlens=(1 3 4 6 8)
max_pathlens=(3 6)
# thresholds=(0.29 0.47 0.49 0.51 0.54 0.62 0.7 0.71 0.666 0.694 0.735)
thresholds=(0.29 0.5 0.51 0.69 0.7 0.74 0.75 0.77 0.78 0.79 0.8 0.82 0.83)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		for threshold in "${thresholds[@]}";
		do
			jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$threshold"_conf_oaei_german_logmap.txt" python Attention_german_amlconf_oaei.py Input/data_conf_oaei_german_logmap_thresh.pkl $max_path $max_pathlen $threshold  "Output/test_conf_oaei_german_logmap_"$max_path"_"$max_pathlen"_"$threshold".pkl" "Models/conf_oaei_german_logmap_"$max_path"_"$max_pathlen"_"$threshold".pt"
		done
	done
done


