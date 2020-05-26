#!/bin/sh


#tools='Query_rewriting Ontology_merging Ritze_2010 Faria_2018 ra1'
#tools='Ritze_2010'

#levenshtein="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
#nbInstSupp="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1 2 3 4 5 6 7 8 9 20 100 cqa_reassess"
#nbInstSupp="0.4"
nbInstSupp="sim"

outputFolder=`pwd`/results/CANARD/

testTypes='cmt-conference cmt-confOf cmt-edas cmt-ekaw conference-cmt conference-confOf conference-edas conference-ekaw confOf-cmt confOf-conference confOf-edas confOf-ekaw edas-cmt edas-conference edas-confOf edas-ekaw ekaw-cmt ekaw-conference ekaw-confOf ekaw-edas'
#testTypes='confOf-ekaw confOf-conference'

#testTypes='confOf-conference'
#for tool in $tools

#for lev in $levenshtein 
for nsupp in $nbInstSupp
do
tool=test_$nsupp
  for testType in $testTypes  
   do
        echo " ---START------ "
        echo "RUNNING $testType WITH $tool"
	toolOutputFolder=$outputFolder/$tool/     
	mkdir -p $toolOutputFolder
        java -jar QueryEvaluator.jar /home/canard/workspace/ComplexAlignmentGenerator/output/conference-query/$tool/$testType $toolOutputFolder
        echo " ---END------ "
        echo "  "
    done
done
