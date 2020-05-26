# Ontology population
5 ontologies from the conference dataset were populated: cmt, conference, confOf, edas, ekaw.
The resulting datasets of the ontology population are in the folder **populated\_dataset**:
- eswc: the ontologies populated with data crawled from the ESWC 2018 website and anonymised.
- data\_100: artifically generated datasets with 0, 20, 40, 60, 80 and 100% of common conference instances.

The population process was articulated in 2 steps:
- creation of the pivot format files (from the website for ESWC, and randomly for the artificial dataset)
- instantiation of SPARQL INSERT query templates with the pivot format files

## Pivot Format (JSON)
You can find examples of conference files in the **data\_files/conferences** folder of this repository.
For each instantiation of the pivot format, there are three files. 
Example with the conf0 instantiation:
 - conf0.json: the file containing the conference event data (registration date, pc members, etc.). It contains the fields *paper\_file* which refers to conf0\_papers.json and *people\_file* which refers to conf0\_people.json
 - conf0\_papers.json: the list of papers submitted to the conference (with title, authors, reviews, reviewers, etc.)
 - conf0\_people.json: the list of people which were used to create the instantiation. (participation to the conference, student, early registration, affiliation, etc.)

## ESWC data extraction (paper data extraction)
To crawl the paper data ESWC 2018 website, the AppExtraction class was run.
```
java -jar AppExtraction.jar [-o <output_file_name>] files...
```
output\_file\_name : "output.json" by default

## Pivot format file generation
To generate conferences data in pivot format, the Generator class can be run.
```
java -jar Generator.jar
```
It outputs 25 different conference data. Each conference has a size randomly chosen between 1 and 10 which will determine for example the number of submitted papers, program committee members, etc.
The generation of this data relies on various endpoints such as DBpedia, scholarlydata, HAL, as well as a set of conference and workshop names from DBLP. All the queries and files are in the **scholarly\_data\_queries** folder of this repository.

## Pivot format to SPARQL INSERT queries 
The populator instantiates SPARQL INSERT query templates with the pivot format conference files.
The SPARQL INSERT query templates are located in the **query\_templates** folder of this repository.

To call the populator:
```
java -jar Populator.jar path/to/parameterfile.json
```
path/to/parameterfile.json is the path to a **JSON** file such containing:
- *name*: the name of the ontology
- *instance\_prefix*: the prefix to give the instances
- *conference\_file\_path*: the list of path to the conf.json files of the ontologies which will be used to populate the ontology.
- *endpoint*: a sparql endpoint on which the ontology is already loaded

Examples of the parameter files can be found in the **parameters** folder of this repository.

# Evaluation system
The evaluation system (Evaluator class) is based on Comptency Questions for Alignment (CQAs).

## CQAs
The Competency Questions for Alignment represent pieces of information.
The ones used in the evaluation are in the **CQAs** folder of this repository.
The coverage of the CQAs by the ontologies is recorded in the **CQA_coverage.ods** file.

## Evaluator
The evaluator can be run like this:
```
java -jar Evaluator.jar path/to/alignment.edoal path/to/result/folder
```
path/to/alignment.edoal is the path to the alignment file. The name of the alignment file must be "source-target.edoal" and the alignment must be in EDOAL format.
The evaluator outputs a "results/source-target.csv" file giving the precision, recall and F-measure score for each CQA covered by both the source and target ontologies.

The evaluator relies on two kinds of query rewriting: one based on (1:n) correspondences, the other based on instances, which can deal with only one (m:n) correspondence per query.
The evaluator will need two SPARQL endpoints with populated data: <http://localhost:3030/source_100> and <http://localhost:3030/target_100>.

For example, when evaluating the cmt-conference.edoal alignment, the endpoints <http://localhost:3030/cmt_100> and <http://localhost:3030/conference_100> must be up and running and must respectively contain the cmt and conference ontologies, both populated with the 100% dataset.

It outputs the following scores for the rewritten target query whose instances are the most similar to those of the reference target CQA.
 - the classical CQA Coverage
 - the recall-oriented CQA Coverage
 - the precision-oriented CQA Coverage
 - the overlap CQA Coverage
 - the query f-measure CQA Coverage


## Precision
The Precision evaluator can be run like this: 
```
java -jar Precision.jar path/to/alignment.edoal 
```
It compares the instances described by the source and target member of each correspondence and prints: 
 - the classical precision
 - the recall-oriented precision
 - the precision-oriented precision
 - the overlap precision
 - the query f-measure precision

## Scripts
The script **full-eval.sh** Calls the Evaluator jar with all the ontology pairs test cases. It is called for a set of tools (or output alignments) that will be evaluated.

The **precision.sh** calls the Precision jar over each set of alignments and aggregates them in a tool\_Precision.csv file.

The script **treat-query-eval.sh** takes the mean value for each pair of ontologies for each tool and aggregates them in a toolname.csv file.


# References
- EDOAL: <http://alignapi.gforge.inria.fr/edoal.html>
- CQA definition: <https://people.kmi.open.ac.uk/francesco/wp-content/uploads/2018/11/EKAWDC2018_5.pdf>
- (1:n) Rewriting system: <https://framagit.org/IRIT_UT2J/sparql-translator-complex-alignment>
