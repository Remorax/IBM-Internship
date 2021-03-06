#!/bin/bash

max_pathlens=(10 12 26 18 20 21 22 24 28 32 38)

for max_pathlen in "${max_pathlens[@]}";
do
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_bagofnbrs.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep.py $max_pathlen Input/data_demarcated_bagofnbrs.pkl "Output/test_demarcated_bagofnbrs"$max_pathlen".pkl"
	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_bagofnbrs_wtpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_weightedpath.py $max_pathlen Input/data_demarcated_bagofnbrs.pkl "Output/test_demarcated_bagofnbrs_wtpath"$max_pathlen".pkl"
done

# max_pathlens=(2)

# for max_pathlen in "${max_pathlens[@]}";
# do
# 	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep.py $max_pathlen Input/data_demarcated.pkl "Output/test_demarcated"$max_pathlen".pkl"
# 	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_wtpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_weightedpath.py $max_pathlen Input/data_demarcated.pkl "Output/test_demarcated_wtpath"$max_pathlen".pkl"
# 	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_uniqpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep.py $max_pathlen Input/data_demarcated_uniqpath.pkl "Output/test_demarcated_uniqpath"$max_pathlen".pkl"
# 	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_wtpath_uniqpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_weightedpath.py $max_pathlen Input/data_demarcated_uniqpath.pkl "Output/test_demarcated_uniqpath_wtpath"$max_pathlen".pkl"
# done

# max_pathlens=(3)

# for max_pathlen in "${max_pathlens[@]}";
# do
# 	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep.py $max_pathlen Input/data_demarcated.pkl "Output/test_demarcated"$max_pathlen".pkl"
# 	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_uniqpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep.py $max_pathlen Input/data_demarcated_uniqpath.pkl "Output/test_demarcated_uniqpath"$max_pathlen".pkl"
# 	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep_wtpath_uniqpath.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_weightedpath.py $max_pathlen Input/data_demarcated_uniqpath.pkl "Output/test_demarcated_uniqpath_wtpath"$max_pathlen".pkl"
# done

# max_pathlens=(4 5)

# for max_pathlen in "${max_pathlens[@]}";
# do
# 	jbsub -q x86_24h -mem 40g -require v100 -cores 1x1+1 -out "Results/Output_att"$max_pathlen"_twostep.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep.py $max_pathlen Input/data_demarcated.pkl "Output/test_demarcated"$max_pathlen".pkl"
# done

