#!/bin/sh


tools='Query_rewriting Ontology_merging ra1 Faria_2018-nora1 Ritze_2010-nora1'
#tools='Ontology_merging'
#tools='Ritze_2010-nora1'

#tools='test_query_reassess test_query'

alignmentFolder=`pwd`/alignments/
outputFolder=`pwd`/results/Precision

testTypes='cmt-conference cmt-confOf cmt-edas cmt-ekaw conference-cmt conference-confOf conference-edas conference-ekaw confOf-cmt confOf-conference confOf-edas confOf-ekaw edas-cmt edas-conference edas-confOf edas-ekaw ekaw-cmt ekaw-conference ekaw-confOf ekaw-edas'

#testTypes='confOf-conference'


for tool in $tools
  do
echo pair,classical,recall oriented,precision oriented,overlap,query f-measure,not disjoint > $outputFolder/$tool"_"Precision.csv
  for testType in $testTypes  
   do
        echo " ---START------ "
        echo "RUNNING $testType WITH $tool"
	toolOutputFolder=$outputFolder/$tool/     
	mkdir -p $toolOutputFolder
	line=`java -jar Precision.jar $alignmentFolder/$tool/$testType.edoal`
        #line=`java -jar Precision.jar /home/canard/workspace/ComplexAlignmentGenerator/output/conference/$tool/$testType.edoal`
	echo $testType,$line >> $outputFolder/$tool"_"Precision.csv
        echo " ---END------ "
        echo "  "
    done
done
