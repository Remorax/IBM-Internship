#!/bin/bash



neighbours=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)



for current in "${neighbours[@]}";
do
        jbsub -q x86_24h -mem 16g -require k80 -cores 1x1+1 -out "Output_att_adjusted"$current".txt" python Attention_oversampled_adjusted.py $current
done
