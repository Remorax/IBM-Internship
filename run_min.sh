#!/bin/bash



neighbours=(1 2 3 4 5 6 7 8)



for current in "${neighbours[@]}";
do
        jbsub -q x86_24h -mem 16g -require k80 -cores 1x1+1 -out "Output_att"$current"_min.txt" python Attention_minlimit.py 24 $current
done
