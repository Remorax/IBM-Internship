#!/bin/bash

max_pathlens=(2 3 4 5 8 10 11 14)
max_paths=(1 2 3 4 5 7 8 10 12 15 18 20 22 24 26 30 32 40 80 152)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_anatomy_aml_bagofnbrs.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_anatomy_aml.py Input/data_anatomy_aml_bagofnbrs.pkl $max_path $max_pathlen  "Output/test_anatomy_aml_bagofnbrs"$max_path"_"$max_pathlen".pkl" "Models/anatomy_aml_bagofnbrs"$max_path"_"$max_pathlen".pt"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_anatomy_aml_bagofnbrs_wtpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_anatomy_aml_weighted.py Input/data_anatomy_aml_bagofnbrs.pkl $max_path $max_pathlen "Output/test_anatomy_aml_bagofnbrs_wtpath"$max_path"_"$max_pathlen".pkl" "Models/anatomy_aml_bagofnbrs_wtpath"$max_path"_"$max_pathlen".pt"
	done
done

max_pathlens=(4 5 8 10)
max_paths=(7 8 10 12 15)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_anatomy_aml_bagofnbrs_wtpath_4props.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_anatomy_aml_4props.py Input/data_anatomy_aml_bagofnbrs.pkl $max_path $max_pathlen "Output/test_anatomy_aml_bagofnbrs_4props"$max_path"_"$max_pathlen".pkl" "Models/anatomy_aml_bagofnbrs_4props"$max_path"_"$max_pathlen".pt"
	done
done

max_pathlens=(3 5 6 7 8 10 13 18 20 25 30 60 152)
max_paths=(1 2 3 4 6 9 11 15 18 22)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_paths in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_anatomy_aml.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_anatomy_aml.py.py Input/data_anatomy_aml.pkl $max_path $max_pathlen "Output/test_anatomy_aml"$max_path"_"$max_pathlen".pkl" "Models/anatomy_aml"$max_path"_"$max_pathlen".pt"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_anatomy_aml_wtpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_anatomy_aml_weighted.py Input/data_anatomy_aml.pkl $max_path $max_pathlen "Output/test_anatomy_aml_wtpath"$max_path"_"$max_pathlen".pkl" "Models/anatomy_aml_wtpath"$max_path"_"$max_pathlen".pt"
	done
done