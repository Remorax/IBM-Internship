#!/bin/bash



neighbours=(24)



for current in "${neighbours[@]}";
do
        jbsub -q x86_24h -mem 16g -require k80 -cores 1x1+1 -out "Output_att_context"$current".txt" python Attention_context.py $current
done
