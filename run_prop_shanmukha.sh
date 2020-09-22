# !/bin/bash

# Train on AML conference + German dataset v2

max_paths=(3 13 21 26)
# max_paths=(3 5 7 12 18 21 24 26)
max_pathlens=(5 8)
# max_pathlens=(1 3 4 6 8)
thresholds=(0.6075)
fns_entity=(0.8 0.85 0.87 0.9 1)
fns_prop=(0.5)

for max_pathlen in "${max_pathlens[@]}";
do
  for max_path in "${max_paths[@]}";
  do
    for threshold in "${thresholds[@]}";
    do
      for fn_entity in "${fns_entity[@]}";
      do
        for fn_prop in "${fns_prop[@]}";
        do
          jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$threshold"_"$fn_entity"_"$fn_prop"_fn_prop.txt" python Attention_twostep_prop.py Input/data_conf_oaei_german_aml_prop_thresh.pkl $max_path $max_pathlen $threshold $fn_entity $fn_prop "Models/conf_oaei_german_aml_"$max_path"_"$max_pathlen"_"$threshold"_"$fn_entity"_"$fn_prop"_fn_prop.pt"
        done
      done
    done
  done
done

max_paths=(3 13 21 26)
# max_paths=(3 5 7 12 18 21 24 26)
max_pathlens=(5 8)
# max_pathlens=(1 3 4 6 8)
thresholds=(0.6075)
fns_entity=(0.35 0.4 0.43)
fns_prop=(0.5)

for max_pathlen in "${max_pathlens[@]}";
do
  for max_path in "${max_paths[@]}";
  do
    for threshold in "${thresholds[@]}";
    do
      for fn_entity in "${fns_entity[@]}";
      do
        for fn_prop in "${fns_prop[@]}";
        do
          jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$threshold"_"$fn_entity"_"$fn_prop"_fn_prop_corrected.txt" python Attention_twostep_prop.py Input/data_conf_oaei_german_aml_corrected.pkl $max_path $max_pathlen $threshold $fn_entity $fn_prop "Models/conf_oaei_german_aml_"$max_path"_"$max_pathlen"_"$threshold"_"$fn_entity"_"$fn_prop"_fn_prop_corrected.pt"
        done
      done
    done
  done
done
