# # !/bin/bash

# # Train on AML conference + German dataset v2

# max_paths=(3 13 21 26)
# max_pathlens=(5 8)
# thresholds=(0.6075 0.612 0.735)

# for max_pathlen in "${max_pathlens[@]}";
# do
# 	for max_path in "${max_paths[@]}";
# 	do
# 		for threshold in "${thresholds[@]}";
# 		do
# 			jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$threshold"_conf_oaei_german_amlv2.txt" python Attention_german_amlconf_oaeiv2.py Input/data_conf_oaei_german_aml_thresh.pkl $max_path $max_pathlen $threshold  "Output/test_conf_oaei_german_aml_"$max_path"_"$max_pathlen"_"$threshold"v2.pkl" "Models/conf_oaei_german_aml_"$max_path"_"$max_pathlen"_"$threshold"v2.pt"
# 		done
# 	done
# done

# Train on AML + LogMap + German dataset v2

max_paths=(3 13 21 26)
max_pathlens=(5 8)
aml_thresholds=(0.6075 0.612 0.735)
logmap_thresholds=(0.29)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		for aml_threshold in "${aml_thresholds[@]}";
		do
			for logmap_threshold in "${logmap_thresholds[@]}";
			do
				jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$aml_threshold"_"$logmap_threshold"_conf_oaei_german_aml_logmapv2.txt" python Attention_german_aml_logmap.py Input/data_conf_oaei_german_logmap_aml_thresh.pkl $max_path $max_pathlen $aml_threshold $logmap_threshold "Output/test_conf_oaei_german_aml_logmap_"$max_path"_"$max_pathlen"_"$aml_threshold"_"$logmap_threshold"v2.pkl" "Models/conf_oaei_german_aml_logmap_"$max_path"_"$max_pathlen"_"$aml_threshold"_"$logmap_threshold"v2.pt"
			done
		done
	done
done
