#!/bin/sh

TESTDIR=`pwd`

ontos='cmt conference confOf edas ekaw'
ontos1='cmt conference confOf edas ekaw'

tools='Query_rewriting Ontology_merging Ritze_2010 Faria_2018 ra1'
#tools='Ontology_merging'
#levenshtein="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
#nbInstSupp="1 2 3 4 5 6 7 8 9 10 20 100"
#nbInstSupp="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1 2 3 4 5 6 7 8 9 10 20 100 cqa_reassess"
#nbInstSupp="0.4"
#nbInstSupp="sim"
#levenshtein="0.0"

#outputFolder=`pwd`/results/CANARD/
outputFolder=`pwd`/results/dataset_100

for tool in $tools
#for lev in $levenshtein 
#for nsupp in $nbInstSupp
  do
#tool=test_$nsupp
echo $tool
#echo pair,equiv,recall oriented, precision oriented,overlap,f-measure > $outputFolder/$tool"_"Precision.csv
echo pair,equiv,recall oriented, precision oriented,overlap,f-measure > $outputFolder/$tool"_"CQA_Coverage.csv
for o1 in $ontos
do
    for o2 in $ontos1
    do
  #      if [ -e $outputFolder/$tool/CQA_coverage/$o1-$o2.csv ];
  #      then
  #          
  #                line=`grep "global mean" $outputFolder/$tool/CQA_coverage/$o1-$o2.csv`
#		line=${line#"global mean,"}
#                 echo $o1-$o2,$line >> $outputFolder/$tool"_"CQA_coverage.csv
#        fi
#	if [ -e $outputFolder/$tool/Precision/$o1-$o2.csv ];
#        then
#            
#                  line=`grep "global mean" $outputFolder/$tool/Precision/$o1-$o2.csv`
#		line=${line#"global mean,"}
#                 echo $o1-$o2,$line >> $outputFolder/$tool"_"Precision.csv
#        fi
	if [ -e $outputFolder/$tool/$o1-$o2.csv ];
        then
            
                  line=`grep "queries" $outputFolder/$tool/$o1-$o2.csv`
		line=${line#"queries,"}
             #    echo $o1-$o2,$line >> $outputFolder/$tool"_"Precision.csv
		line=`grep "CQAs" $outputFolder/$tool/$o1-$o2.csv`
		line=${line#"CQAs,"}
                 echo $o1-$o2,$line >> $outputFolder/$tool"_"CQA_Coverage.csv
        fi

    done
done
done

