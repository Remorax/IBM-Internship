# !/bin/bash

# Train on AML conference + German dataset 

max_pathlens=(12 13 21 49)
max_paths=(2)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_german_aml_bagofnbrs.txt" python Attention_german_amlconf_oaei.py Input/data_conf_oaei_german_aml_bagofnbrs.pkl $max_path $max_pathlen  "Output/test_conf_oaei_german_aml_bagofnbrs"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_german_aml_bagofnbrs"$max_path"_"$max_pathlen".pt"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_german_aml_bagofnbrs_wtpath.txt" python Attention_german_amlconf_oaei_weighted.py Input/data_conf_oaei_german_aml_bagofnbrs.pkl $max_path $max_pathlen "Output/test_conf_oaei_german_aml_wtpath_bagofnbrs"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_german_aml_wtpath_bagofnbrs"$max_path"_"$max_pathlen".pt"
	done
done

max_pathlens=(3 4 5 8)
max_paths=(1 3 4 8 13 21 49)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_german_aml.txt" python Attention_german_amlconf_oaei.py Input/data_conf_oaei_german_aml.pkl $max_path $max_pathlen  "Output/test_conf_oaei_german_aml"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_german_aml"$max_path"_"$max_pathlen".pt"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_german_aml_wtpath.txt" python Attention_german_amlconf_oaei_weighted.py Input/data_conf_oaei_german_aml.pkl $max_path $max_pathlen "Output/test_conf_oaei_german_aml_wtpath"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_german_aml_wtpath"$max_path"_"$max_pathlen".pt"
	done
done

