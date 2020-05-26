#!/bin/sh

TESTDIR=`pwd`

ontos='cmt conference confOf edas ekaw'
ontos1='cmt conference confOf edas ekaw'


levenshtein="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1.0"
#levenshtein="0.0"

timeFolder=/home/canard/workspace/ComplexAlignmentGenerator/output/time
outputFolder=`pwd`/results/CANARD

for lev in $levenshtein 
 do
echo "threshold, real time" > $outputFolder/time_sim.csv
time=0
echo $lev
for o1 in $ontos
do
    for o2 in $ontos1
    do
        if [ $o1 != $o2 ]
        then
             
                line=`grep "real" $timeFolder"/time_"$o1-$o2"_"sim.txt`
		line=${line#"real "}
		line=${line%.*}
		time=$(($time+$line))
        fi
    done
done
echo $time
 echo $lev,$time >> $outputFolder/time_sim.csv
done

