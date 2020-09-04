# !/bin/bash

# Train on AML conference + German dataset 

max_pathlens=(4 6 12 13 21 49)
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
	for max_paths in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_german_aml.txt" python Attention_german_amlconf_oaei.py Input/data_conf_oaei_german_aml.pkl $max_path $max_pathlen  "Output/test_conf_oaei_german_aml"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_german_aml"$max_path"_"$max_pathlen".pt"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_german_aml_wtpath.txt" python Attention_german_amlconf_oaei_weighted.py Input/data_conf_oaei_german_aml.pkl $max_path $max_pathlen "Output/test_conf_oaei_german_aml_wtpath"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_german_aml_wtpath"$max_path"_"$max_pathlen".pt"
	done
done

# Train on AML conference 

max_pathlens=(2 3 4 8 10 13 38)
max_paths=(2)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_paths in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_aml_bagofnbrs.txt" python Attention_twostep_aml.py Input/data_conf_oaei_aml_bagofnbrs.pkl $max_path $max_pathlen  "Output/test_conf_oaei_aml_bagofnbrs"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_aml_bagofnbrs"$max_path"_"$max_pathlen".pt"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_aml_wtpath_bagofnbrs.txt" python Attention_twostep_aml_wtpath.py Input/data_conf_oaei_aml_bagofnbrs.pkl $max_path $max_pathlen "Output/test_conf_oaei_aml_wtpath_bagofnbrs"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_aml_wtpath_bagofnbrs"$max_path"_"$max_pathlen".pt"
	done
done

max_pathlens=(2 3 4 7)
max_paths=(2 3 5 6 10 22)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_paths in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_aml.txt" python Attention_twostep_aml.py Input/data_conf_oaei_aml.pkl $max_path $max_pathlen  "Output/test_conf_oaei_aml"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_aml"$max_path"_"$max_pathlen".pt"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_aml_wtpath.txt" python Attention_twostep_aml_wtpath.py Input/data_conf_oaei_aml.pkl $max_path $max_pathlen "Output/test_conf_oaei_aml_wtpath"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_aml_wtpath"$max_path"_"$max_pathlen".pt"
	done
done

# Train on German dataset

max_pathlens=(2 4 5 6 12 21 49)
max_paths=(2)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_paths in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_german_bagofnbrs.txt" python Attention_twostep_german_oaei.py Input/data_conf_oaei_german_bagofnbrs.pkl $max_path $max_pathlen  "Output/test_conf_oaei_german_bagofnbrs"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_german_bagofnbrs"$max_path"_"$max_pathlen".pt"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_german_wtpath_bagofnbrs.txt" python Attention_twostep_german_wtpath_oaei.py Input/data_conf_oaei_german_bagofnbrs.pkl $max_path $max_pathlen "Output/test_conf_oaei_german_wtpath_bagofnbrs"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_german_wtpath_bagofnbrs"$max_path"_"$max_pathlen".pt"
	done
done

max_pathlens=(3 4 5 6 8)
max_paths=(2 4 9 13 17 22 49)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_paths in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_german.txt" python Attention_twostep_german_oaei.py Input/data_conf_oaei_german.pkl $max_path $max_pathlen  "Output/test_conf_oaei_german"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_german"$max_path"_"$max_pathlen".pt"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_conf_oaei_german_wtpath.txt" python Attention_twostep_german_wtpath_oaei.py Input/data_conf_oaei_german.pkl $max_path $max_pathlen "Output/test_conf_oaei_german_wtpath"$max_path"_"$max_pathlen".pkl" "Models/conf_oaei_german_wtpath"$max_path"_"$max_pathlen".pt"
	done
done