# !/bin/bash

max_pathlens=(1 7 9 12 14 16 17 18 21 23 24 25 26 29 38)
max_paths=(1 2)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_conf_bagofnbrs_part1.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_part1.py Input/data_conf_bagofnbrs.pkl $max_path $max_pathlen "Models/conf_bagofnbrs"$max_path"_"$max_pathlen".pt" "Output/conf_bagofnbrs"$max_path"_"$max_pathlen"_part1.pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_conf_bagofnbrs_part2.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_part2.py Input/data_conf_bagofnbrs.pkl $max_path $max_pathlen "Models/conf_bagofnbrs"$max_path"_"$max_pathlen".pt" "Output/conf_bagofnbrs"$max_path"_"$max_pathlen"_part2.pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_weighted_conf_bagofnbrs_part1.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_weightedpath_part1.py Input/data_conf_bagofnbrs.pkl $max_path $max_pathlen "Models/conf_weighted_bagofnbrs"$max_path"_"$max_pathlen".pt" "Output/conf_weighted_bagofnbrs"$max_path"_"$max_pathlen"_part1.pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_weighted_conf_bagofnbrs_part2.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_weightedpath_part2.py Input/data_conf_bagofnbrs.pkl $max_path $max_pathlen "Models/conf_weighted_bagofnbrs"$max_path"_"$max_pathlen".pt" "Output/conf_weighted_bagofnbrs"$max_path"_"$max_pathlen"_part2.pkl"
	done
done

max_pathlens=(1 2 3 5 6 8 9 10 21 22)
max_paths=(1 5 7)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_conf_part1.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_part1.py Input/data_conf.pkl $max_path $max_pathlen "Models/conf"$max_path"_"$max_pathlen".pt" "Output/conf"$max_path"_"$max_pathlen"_part1.pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_conf_part2.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_part2.py Input/data_conf.pkl $max_path $max_pathlen "Models/conf"$max_path"_"$max_pathlen".pt" "Output/conf"$max_path"_"$max_pathlen"_part2.pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_weighted_conf_part1.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_weightedpath_part1.py Input/data_conf.pkl $max_path $max_pathlen "Models/conf_weighted"$max_path"_"$max_pathlen".pt" "Output/conf_weighted"$max_path"_"$max_pathlen"_part1.pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_weighted_conf_part2.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_weightedpath_part2.py Input/data_conf.pkl $max_path $max_pathlen "Models/conf_weighted"$max_path"_"$max_pathlen".pt" "Output/conf_weighted"$max_path"_"$max_pathlen"_part2.pkl"
	done
done

max_pathlens=(1 9 10 22)
max_paths=(2 3 4 6)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_conf_part1.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_part1.py Input/data_conf.pkl $max_path $max_pathlen "Models/conf"$max_path"_"$max_pathlen".pt" "Output/conf"$max_path"_"$max_pathlen"_part1.pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_conf_part2.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_part2.py Input/data_conf.pkl $max_path $max_pathlen "Models/conf"$max_path"_"$max_pathlen".pt" "Output/conf"$max_path"_"$max_pathlen"_part2.pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_weighted_conf_part1.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_weightedpath_part1.py Input/data_conf.pkl $max_path $max_pathlen "Models/conf_weighted"$max_path"_"$max_pathlen".pt" "Output/conf_weighted"$max_path"_"$max_pathlen"_part1.pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_weighted_conf_part2.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_conf_twostep_weightedpath_part2.py Input/data_conf.pkl $max_path $max_pathlen "Models/conf_weighted"$max_path"_"$max_pathlen".pt" "Output/conf_weighted"$max_path"_"$max_pathlen"_part2.pkl"
	done
done