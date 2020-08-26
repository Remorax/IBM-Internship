#!/bin/bash

max_pathlens=(10 12 26 18 20 21 22 24 28 32 38)

for max_pathlen in "${max_pathlens[@]}";
do
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_aml_bagofnbrs.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_aml.py $max_pathlen Input/data_aml_bagofnbrs.pkl "Output/test_aml_bagofnbrs"$max_pathlen".pkl" "Models/aml_bagofnbrs"$max_pathlen".pkl"
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_aml_bagofnbrs_wtpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_aml_wtpath.py $max_pathlen Input/data_aml_bagofnbrs.pkl "Output/test_aml_bagofnbrs_wtpath"$max_pathlen".pkl" "Models/aml_bagofnbrs_wtpath"$max_pathlen".pkl"
done

max_pathlens=(2 3 4 5)

for max_pathlen in "${max_pathlens[@]}";
do
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_aml.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_aml.py $max_pathlen Input/data_aml.pkl "Output/test_aml"$max_pathlen".pkl" "Models/aml"$max_pathlen".pkl"
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_aml_wtpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_aml_wtpath.py $max_pathlen Input/data_aml.pkl "Output/test_aml_wtpath"$max_pathlen".pkl" "Models/aml_wtpath"$max_pathlen".pkl"
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_aml_uniqpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_aml.py $max_pathlen Input/data_aml_uniqpath.pkl "Output/test_aml_uniqpath"$max_pathlen".pkl" "Models/aml_uniqpath"$max_pathlen".pkl"
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_aml_wtpath_uniqpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_aml_wtpath.py $max_pathlen Input/data_aml_uniqpath.pkl "Output/test_aml_uniqpath_wtpath"$max_pathlen".pkl" "Models/aml_uniqpath_wtpath"$max_pathlen".pkl"
done
