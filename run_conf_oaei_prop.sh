# !/bin/bash

# Train on AML conference + German dataset v2

max_paths=(3 13 21 26)
# max_paths=(3 5 7 12 18 21 24 26)
max_pathlens=(5 8)
# max_pathlens=(1 3 4 6 8)
thresholds=(0.6075 0.612)

for max_pathlen in "${max_pathlens[@]}";
do
  for max_path in "${max_paths[@]}";
  do
      for threshold in "${thresholds[@]}";
      do
          jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "Results/Output_att"$max_path"_"$max_pathlen"_"$threshold"_conf_oaei_german_amlv2_prop.txt" python Attention_twostep_prop.py Input/data_conf_oaei_german_aml_prop_thresh.pkl $max_path $max_pathlen $threshold  "Output/test_conf_oaei_german_aml_prop_"$max_path"_"$max_pathlen"_"$threshold"v2.pkl" "Models/conf_oaei_german_aml_"$max_path"_"$max_pathlen"_"$threshold"v2_prop.pt"
      done
  done
done
