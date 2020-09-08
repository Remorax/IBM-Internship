# !/bin/bash

# 193 jobs

# Lebensmittel 

# max_paths=(1 2 6 9 16)
# max_pathlens=(1 3 4 5 7)

# for max_pathlen in "${max_pathlens[@]}";
# do
# 	for max_path in "${max_paths[@]}";
# 	do
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_lebensmittelv2.txt" python Attention_twostep_german.py Input/data_lebensmittel.pkl $max_path $max_pathlen "Output/test"$max_path"_"$max_pathlen"_twostep_lebensmittelv2.pkl"
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_lebensmittel_weightedv2.txt" python Attention_twostep_german_weighted.py Input/data_lebensmittel.pkl $max_path $max_pathlen "Output/test"$max_path"_"$max_pathlen"_twostep_lebensmittelv2_weighted.pkl"
# 	done
# done

# max_paths=(1)
# max_pathlens=(1 3 5 6 7 9 16)

# for max_pathlen in "${max_pathlens[@]}";
# do
# 	for max_path in "${max_paths[@]}";
# 	do
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_lebensmittel_bagofnbrsv2.txt" python Attention_twostep_german.py Input/data_lebensmittel_bagofnbrs.pkl $max_path $max_pathlen "Output/test"$max_path"_"$max_pathlen"_twostep_lebensmittelv2_bagofnbrs.pkl"
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_lebensmittel_weighted_bagofnbrsv2.txt" python Attention_twostep_german_weighted.py Input/data_lebensmittel_bagofnbrs.pkl $max_path $max_pathlen "Output/test"$max_path"_"$max_pathlen"_twostep_lebensmittelv2_weighted_bagofnbrs.pkl"
# 	done
# done

# # Freizeit

# max_paths=(1 2 3 4 7 12)
# max_pathlens=(1 3 4 6)

# for max_pathlen in "${max_pathlens[@]}";
# do
# 	for max_path in "${max_paths[@]}";
# 	do
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_freizeitv2.txt" python Attention_twostep_german.py Input/data_freizeit.pkl $max_path $max_pathlen "Output/test"$max_path"_"$max_pathlen"_twostep_freizeitv2.pkl"
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_freizeit_weightedv2.txt" python Attention_twostep_german_weighted.py Input/data_freizeit.pkl $max_path $max_pathlen "Output/test"$max_path"_"$max_pathlen"_twostep_freizeitv2_weighted.pkl"
# 	done
# done

# max_paths=(1)
# max_pathlens=(1 3 4 5 7 9 12)

# for max_pathlen in "${max_pathlens[@]}";
# do
# 	for max_path in "${max_paths[@]}";
# 	do
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_freizeit_bagofnbrsv2.txt" python Attention_twostep_german.py Input/data_freizeit_bagofnbrs.pkl $max_path $max_pathlen "Output/test"$max_path"_"$max_pathlen"_twostep_freizeitv2_bagofnbrs.pkl"
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_freizeit_weighted_bagofnbrsv2.txt" python Attention_twostep_german_weighted.py Input/data_freizeit_bagofnbrs.pkl $max_path $max_pathlen "Output/test"$max_path"_"$max_pathlen"_twostep_freizeitv2_bagofnbrs_weighted.pkl"
# 	done
# done

# Web directory

# max_paths=(1 2 3 4 6 9 13 21 49)
# max_pathlens=(1 3 4 5 6 8)

# for max_pathlen in "${max_pathlens[@]}";
# do
# 	for max_path in "${max_paths[@]}";
# 	do
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_webdir.txt" python Attention_twostep_german.py ../data_webdir.pkl $max_path $max_pathlen
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_webdir_weighted.txt" python Attention_twostep_german_weighted.py ../data_webdir.pkl $max_path $max_pathlen
# 	done
# done

# max_paths=(1)
# max_pathlens=(1 3 4 5 6 8 9 13 20 21 49)

# for max_pathlen in "${max_pathlens[@]}";
# do
# 	for max_path in "${max_paths[@]}";
# 	do
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_webdir_bagofnbrs.txt" python Attention_twostep_german.py ../data_webdir_bagofnbrs.pkl $max_path $max_pathlen
# 		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_webdir_weighted_bagofnbrs.txt" python Attention_twostep_german_weighted.py ../data_webdir_bagofnbrs.pkl $max_path $max_pathlen
# 	done
# done

# Hybrid webdir + lebensmittel

max_paths=(1 2 3 4 6 9 13 15 21 49)
max_pathlens=(1 3 4 5 6 8)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_hybrid_leb.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_hybrid.py ../data_webdir_leb.pkl $max_path $max_pathlen "Models/twostep_hybrid_leb"$max_path"_"$max_pathlen".pt" "Output/twostep_hybrid_leb"$max_path"_"$max_pathlen".pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_hybrid_leb_weighted.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_hybrid_weighted.py ../data_webdir_leb.pkl $max_path $max_pathlen "Models/twostep_hybrid_leb_weighted"$max_path"_"$max_pathlen".pt" "Output/twostep_hybrid_leb_weighted"$max_path"_"$max_pathlen".pkl"
	done
done

max_paths=(1)
max_pathlens=(1 3 4 5 6 7 8 13 15 21 49)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_hybrid_leb_bagofnbrs.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_hybrid.py ../data_webdir_leb_bagofnbrs.pkl $max_path $max_pathlen "Models/twostep_hybrid_leb_bagofnbrs"$max_path"_"$max_pathlen".pt" "Output/twostep_hybrid_leb_bagofnbrs"$max_path"_"$max_pathlen".pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_hybrid_leb_weighted_bagofnbrs.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_hybrid_weighted.py ../data_webdir_leb_bagofnbrs.pkl $max_path $max_pathlen "Models/twostep_hybrid_leb_weighted_bagofnbrs"$max_path"_"$max_pathlen".pt" "Output/twostep_hybrid_leb_weighted_bagofnbrs"$max_path"_"$max_pathlen".pkl"
	done
done

# Hybrid webdir + freizeit

max_paths=(1 2 3 4 6 7 9 13 21 49)
max_pathlens=(1 3 4 5 6 8)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_hybrid_fre.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_hybrid.py ../data_webdir_fre.pkl $max_path $max_pathlen "Models/twostep_hybrid_fre"$max_path"_"$max_pathlen".pt" "Output/twostep_hybrid_fre"$max_path"_"$max_pathlen".pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_hybrid_fre_weighted.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_hybrid_weighted.py ../data_webdir_fre.pkl $max_path $max_pathlen "Models/twostep_hybrid_fre_weighted"$max_path"_"$max_pathlen".pt" "Output/twostep_hybrid_fre_weighted"$max_path"_"$max_pathlen".pkl"
	done
done

max_paths=(1)
max_pathlens=(1 3 4 5 6 7 8 12 15 21 49)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_hybrid_fre_bagofnbrs.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_hybrid.py ../data_webdir_fre_bagofnbrs.pkl $max_path $max_pathlen "Models/twostep_hybrid_fre_bagofnbrs"$max_path"_"$max_pathlen".pt" "Output/twostep_hybrid_fre_bagofnbrs"$max_path"_"$max_pathlen".pkl"
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_twostep_hybrid_fre_weighted_bagofnbrs.txt" ~/anaconda2/envs/myenv/bin/python3.6 Attention_twostep_german_hybrid_weighted.py ../data_webdir_fre_bagofnbrs.pkl $max_path $max_pathlen "Models/twostep_hybrid_fre_weighted_bagofnbrs"$max_path"_"$max_pathlen".pt" "Output/twostep_hybrid_fre_weighted_bagofnbrs"$max_path"_"$max_pathlen".pkl"
	done
done