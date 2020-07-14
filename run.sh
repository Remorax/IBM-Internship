#!/bin/bash



neighbours=(3 5 8 10 12 18 20 21 22 23 24)



for current in "${neighbours[@]}";
do
        jbsub -q x86_24h -mem 16g -require k80 -cores 1x1+1 -out "Output_att"$current".txt" python Attention.py $current
done
