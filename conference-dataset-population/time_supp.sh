#!/bin/sh

TESTDIR=`pwd`

ontos='cmt conference confOf edas ekaw'
ontos1='cmt conference confOf edas ekaw'


#levenshtein="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
#levenshtein="0.0"
support="1 2 3 4 5 6 7 8 9 10 20 100"
#support="query"

timeFolder=/home/canard/workspace/ComplexAlignmentGenerator/output/time
outputFolder=`pwd`/results/CANARD
echo "threshold, time" > $outputFolder/time_supp.csv
for supp in $support 
  do

time=0
echo $supp
for o1 in $ontos
do
    for o2 in $ontos1
    do
        if [ $o1 != $o2 ]
        then
             
                line=`grep "real" $timeFolder"/time_"$o1-$o2"_"$supp"_0.4.txt"`
		line=${line#"real "}
		line=${line%.*}
		time=$(($time+$line))
        fi
    done
done
echo $time
 echo $supp,$time >> $outputFolder/time_supp.csv
done

