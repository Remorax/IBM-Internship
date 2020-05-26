#!/bin/sh


tools='Query_rewriting Ontology_merging ra1 Ritze_2010 Faria_2018'
tools='LogMap'
#tools='Ritze_2010 Faria_2018'
#tools='Query_rewriting Ontology_merging'
#tools='test_query_reassess test_query'

alignmentFolder=`pwd`/alignments
#outputFolder=`pwd`/results/CANARD
outputFolder=`pwd`/results/dataset_100



testTypes='cmt-conference cmt-confOf cmt-edas cmt-ekaw conference-cmt conference-confOf conference-edas conference-ekaw confOf-cmt confOf-conference confOf-edas confOf-ekaw edas-cmt edas-conference edas-confOf edas-ekaw ekaw-cmt ekaw-conference ekaw-confOf ekaw-edas'
testTypes='cmt-conference'
#testTypes='confOf-conference'


for tool in $tools
  do
  for testType in $testTypes  
   do
	#tool=test_$lev
        echo " ---START------ "
        echo "RUNNING $testType WITH $tool"
	toolOutputFolder=$outputFolder/$tool/     
	mkdir -p $toolOutputFolder
	java -jar Evaluator.jar $alignmentFolder/$tool/$testType.edoal $toolOutputFolder
        #java -jar Evaluator.jar /home/canard/workspace/ComplexAlignmentGenerator/output/conference/$tool/$testType.edoal $toolOutputFolder
        echo " ---END------ "
        echo "  "
    done
done
