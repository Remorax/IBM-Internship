# !/bin/bash

max_pathlens=(3 4 8 13 20 60 80 152)
max_paths=(1 2 3 6 9 15 22)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_paths in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_anatomy_aml.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_anatomy_aml.py.py Input/data_anatomy_oaei.pkl $max_path $max_pathlen "Output/test_anatomy_aml"$max_path"_"$max_pathlen".pkl" "Models/anatomy_aml"$max_path"_"$max_pathlen".pt"
		jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_anatomy_aml_wtpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_anatomy_aml_weighted.py Input/data_anatomy_oaei.pkl $max_path $max_pathlen "Output/test_anatomy_aml_wtpath"$max_path"_"$max_pathlen".pkl" "Models/anatomy_aml_wtpath"$max_path"_"$max_pathlen".pt"
	done
done