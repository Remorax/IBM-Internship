#!/bin/sh

TESTDIR=`pwd`

ontos='cmt conference confOf edas ekaw'
ontos1='cmt conference confOf edas ekaw'

#tools='Query_rewriting Ontology_merging Ritze_2010 Faria_2018 ra1'

outputFolder=`pwd`/results/dataset_100

for tool in $tools
do
echo $tool
echo pair,precision,f-measure,recall > $outputFolder/$tool.csv
for o1 in $ontos
do
    for o2 in $ontos1
    do
        if [ -e $outputFolder/$tool/$o1-$o2.csv ];
        then
             
            	
                 line=`grep "global mean" $outputFolder/$tool/$o1-$o2.csv`
		line=${line#"global mean,"}
                 echo $o1-$o2,$line >> $outputFolder/$tool.csv
        fi
    done
done
done

