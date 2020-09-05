# !/bin/bash

# Train on AML conference + German dataset 

max_pathlens=(4 6 12 13 21 49)
max_paths=(2)
thresholds=(0.66 0.735 0.736 0.8 0.96 0.98 0.99)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$threshold"_conf_oaei_german_aml_bagofnbrs.txt" python Attention_german_amlconf_oaei.py Input/data_conf_oaei_german_aml_bagofnbrs_thresh.pkl $max_path $max_pathlen $threshold  "Output/test_conf_oaei_german_aml_bagofnbrs"$max_path"_"$max_pathlen"_"$threshold".pkl" "Models/conf_oaei_german_aml_bagofnbrs"$max_path"_"$max_pathlen"_"$threshold".pt"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$threshold"_conf_oaei_german_aml_bagofnbrs_wtpath.txt" python Attention_german_amlconf_oaei_weighted.py Input/data_conf_oaei_german_aml_bagofnbrs_thresh.pkl $max_path $max_pathlen $threshold "Output/test_conf_oaei_german_aml_wtpath_bagofnbrs"$max_path"_"$max_pathlen"_"$threshold".pkl" "Models/conf_oaei_german_aml_wtpath_bagofnbrs"$max_path"_"$max_pathlen"_"$threshold".pt"
	done
done

max_pathlens=(4 5 8)
max_paths=(13 21 49)
thresholds=(0.66 0.735 0.736 0.8 0.96 0.98)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		for threshold in "${thresholds[@]}";
		do
			jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$threshold"_conf_oaei_german_aml.txt" python Attention_german_amlconf_oaei.py Input/data_conf_oaei_german_aml_thresh.pkl $max_path $max_pathlen $threshold  "Output/test_conf_oaei_german_aml"$max_path"_"$max_pathlen"_"$threshold".pkl" "Models/conf_oaei_german_aml"$max_path"_"$max_pathlen"_"$threshold".pt"
			jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$threshold"_conf_oaei_german_aml_wtpath.txt" python Attention_german_amlconf_oaei_weighted.py Input/data_conf_oaei_german_aml_thresh.pkl $max_path $max_pathlen $threshold "Output/test_conf_oaei_german_aml_wtpath"$max_path"_"$max_pathlen"_"$threshold".pkl" "Models/conf_oaei_german_aml_wtpath"$max_path"_"$max_pathlen"_"$threshold".pt"
		done
	done
done

