# !/bin/bash

# Train on AML conference + German dataset 

max_paths=(13 21 24 26)
max_pathlens=(1)
thresholds=(0.6075 0.612)

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

max_paths=(13 24 26)
max_pathlens=(1)
thresholds=(0.735)

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

max_paths=(13 21 24 26)
max_pathlens=(3)
thresholds=(0.6075 0.612 0.735)

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

max_paths=(13 18 21 24 26)
max_pathlens=(4)
thresholds=(0.6075 0.612)

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

max_paths=(7 18 21 24 26)
max_pathlens=(4)
thresholds=(0.735)

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


max_paths=(3 5 7 13 18 21 24 26)
max_pathlens=(5 8)
thresholds=(0.6075 0.612 0.735)

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

max_paths=(3 5 13 21 24 26)
max_pathlens=(6)
thresholds=(0.6075 0.612 0.735)

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