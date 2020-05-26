 export CLASSPATH=lib/align.jar:lib/procalign.jar:lib/jena/jena.jar:lib/jena/arq.jar:lib/iddl/iddl.jar:lib/hermit/hermit.jar:lib/ontosim/ontosim.jar:lib/slf4j/slf4j-api.jar:lib/slf4j/jcl-over-slf4j.jar:lib/slf4j/log4j-over-slf4j.jar:lib/xerces/xercesImpl.jar:lib/jena/iri.jar:lib/jena/httpcore.jar:results

#o1List='cmt conference confOf edas'
#o2List='conference confOf edas ekaw'
o1List='confOf'
o2List='edas'


for o1 in $o1List
do
  for o2 in $o2List
do
 java -cp $CLASSPATH fr.inrialpes.exmo.align.cli.ParserPrinter file:EDOAL_alignments/$o1-$o2.edoal -r fr.inrialpes.exmo.align.impl.renderer.SPARQLConstructRendererVisitor -o construct/$o1-$o2.sparql

done
done
