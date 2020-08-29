#!/bin/bash

max_pathlens=(10 12 26 18 20 21 22 24 28 32 38 40 42 44 48 49)

for max_pathlen in "${max_pathlens[@]}";
do
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_german_oaei_bagofnbrs.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_oaei.py $max_pathlen Input/data_german_oaei_bagofnbrs.pkl "Output/test_german_oaei_bagofnbrs"$max_pathlen".pkl" "Models/german_oaei_bagofnbrs"$max_pathlen".pkl"
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_german_oaei_bagofnbrs_wtpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_wtpath_oaei.py $max_pathlen Input/data_german_oaei_bagofnbrs.pkl "Output/test_german_oaei_bagofnbrs_wtpath"$max_pathlen".pkl" "Models/german_oaei_bagofnbrs_wtpath"$max_pathlen".pkl"
done

max_pathlens=(2 3 4 5 6 7)

for max_pathlen in "${max_pathlens[@]}";
do
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_german_oaei.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_oaei.py $max_pathlen Input/data_german_oaei.pkl "Output/test_german_oaei"$max_pathlen".pkl" "Models/german_oaei"$max_pathlen".pkl"
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_german_oaei_wtpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_wtpath_oaei.py $max_pathlen Input/data_german_oaei.pkl "Output/test_german_oaei_wtpath"$max_pathlen".pkl" "Models/german_oaei_wtpath"$max_pathlen".pkl"
done
