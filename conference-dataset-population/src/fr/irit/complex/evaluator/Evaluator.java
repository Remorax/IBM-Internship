package fr.irit.complex.evaluator;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.semanticweb.owl.align.AlignmentVisitor;

import com.fasterxml.jackson.databind.JsonNode;

import fr.inrialpes.exmo.align.impl.BasicAlignment;
import fr.inrialpes.exmo.align.impl.edoal.EDOALAlignment;
import fr.inrialpes.exmo.align.impl.renderer.RDFRendererVisitor;
import fr.inrialpes.exmo.align.parser.AlignmentParser;
import fr.irit.melodi.sparql.proxy.SparqlProxy;

public class Evaluator {

	public static void main(String[] args) {
		try {
			File alignmentfile = new File(args[0]);
			String resultfile = "results/";
			if(args.length>1) {
				resultfile=args[1];
			}
			AlignmentParser ap=new AlignmentParser();
			BasicAlignment al = new BasicAlignment();

			String sourceOnto ;
			String targetOnto ;

			ArrayList<Double> precisions = new ArrayList<Double>();
			ArrayList<Double> recalls  = new ArrayList<Double>();
			ArrayList<Double> fMeasures = new ArrayList<Double>();
			//for each source and target query, get all answers
			ArrayList<HashSet<String>> sourceResults = new ArrayList<HashSet<String>>();
			ArrayList<String> targetQueries = new ArrayList<String>();
			ArrayList<String> sourceQueries = new ArrayList<String>();
			ArrayList<String> rewrittenQueries = new ArrayList<String>();

			if (ap.parse(alignmentfile.toURI()) instanceof EDOALAlignment) {
				al=(EDOALAlignment) ap.parse(alignmentfile.toURI());
				sourceOnto = alignmentfile.getName().replaceAll("-[A-Za-z]+.edoal", "");
				targetOnto = alignmentfile.getName().replaceAll("[A-Za-z]+-", "").replaceAll(".edoal", "");

				//If inverse alignment file does not exist, invert the alignment and create it
				File inverseAlignmentFile = new File(alignmentfile.getPath().replaceAll(sourceOnto+"-"+targetOnto+".edoal", targetOnto+"-"+sourceOnto+".edoal"));
				if(!inverseAlignmentFile.exists()) {
					EDOALAlignment inverseAl = (EDOALAlignment) al.inverse();
					PrintWriter writer = new PrintWriter (inverseAlignmentFile); 
					AlignmentVisitor renderer = new RDFRendererVisitor(writer); 
					inverseAl.render(renderer); 
					writer.flush(); 
					writer.close();
				}

				System.out.println("Translating alignment into SPARQL queries");
				targetQueries = ((EDOALAlignment) al).toTargetSPARQLQuery();
				sourceQueries = ((EDOALAlignment) al).toSourceSPARQLQuery();


				System.out.println("Retrieving Source Alignment SPARQL queries results");

				for(int i = 0; i< targetQueries.size();i++) {
					//System.out.println(targetQueries.get(i));
					//System.out.println(sourceQueries.get(i));
					sourceResults.add(getSPARQLQueryResults(sourceOnto,sourceQueries.get(i)));
				}

			}
			else {
				al= (BasicAlignment) ap.parse(alignmentfile.toURI());
				sourceOnto = alignmentfile.getName().replaceAll("-[A-Za-z]+.rdf", "");
				targetOnto = alignmentfile.getName().replaceAll("[A-Za-z]+-", "").replaceAll(".rdf", "");
				File inverseAlignmentFile = new File(alignmentfile.getPath().replaceAll(sourceOnto+"-"+targetOnto+".rdf", targetOnto+"-"+sourceOnto+".rdf"));
				if(!inverseAlignmentFile.exists()) {
					BasicAlignment inverseAl = (BasicAlignment) al.inverse();
					PrintWriter writer = new PrintWriter (inverseAlignmentFile); 
					AlignmentVisitor renderer = new RDFRendererVisitor(writer); 
					inverseAl.render(renderer); 
					writer.flush(); 
					writer.close();
				}

			}


			String resultCSV = "cqa,best_q_prec,best_q_fmeasure,best_q_rec\n";

			//for each CQA, 
			File queriesFolder = new File("CQAs");
			File[] listOfCQA = queriesFolder.listFiles();
			System.out.println("Retrieving CQAs results");
			for (int i = 0; i < listOfCQA.length; i++) {
				String cqa = listOfCQA[i].getName();
				File sourceCQAFile = new File(listOfCQA[i].getPath()+"/"+sourceOnto+".sparql");
				File targetCQAFile = new File(listOfCQA[i].getPath()+"/"+targetOnto+".sparql");
				//Evaluate CQA only if the 2 exist
				if (sourceCQAFile.exists() && targetCQAFile.exists()) {
					System.out.println(cqa);
					String sourceCQA = getQueryContent(sourceCQAFile);
					String targetCQA = getQueryContent(targetCQAFile);
					HashSet<String> sourceCQAresults = getSPARQLQueryResults(sourceOnto,sourceCQA);
					HashSet<String> targetCQAresults = getSPARQLQueryResults(targetOnto,targetCQA);

					
					if(al instanceof EDOALAlignment) {
						// try to rewrite using 1:n correspondences ==> 1st set of target queries
						rewrittenQueries = ((EDOALAlignment) al).rewriteAllPossibilitiesQuery(sourceCQA);	
						// second list of queries (instance based)
						// compare CQA answers with source query answers
						for(int j=0; j < sourceResults.size();j++) {
							// if identical, add associated targetQuery to the queries
							if (identical(compareHashSet(sourceResults.get(j), sourceCQAresults))) {
								rewrittenQueries.add(targetQueries.get(j));
							}
						}
					}
					else {
						rewrittenQueries.add(al.rewriteSPARQLQuery(sourceCQA));
					}


					//evaluation
					// get query results for the two target query sets
					//compare results of target CQA and target queries from set 1 and 2
					ArrayList<Double> bestRes = new ArrayList<Double>();
					bestRes.add(0.0);
					bestRes.add(0.0);
					double bestFmeasure = 0.0;
					//System.out.println("Retrieving Rewritten queries results");
					for (String rewrittenQuery : rewrittenQueries) {
						//	System.out.println(rewrittenQuery);
						HashSet<String> rewrittenResults = getSPARQLQueryResults(targetOnto,rewrittenQuery);
						ArrayList<Double> rewrittenRes = compareHashSet(targetCQAresults,rewrittenResults);
						if (fMeasure(rewrittenRes) > bestFmeasure) {
							bestRes = rewrittenRes;
							bestFmeasure=fMeasure(rewrittenRes);
						}
					}
					precisions.add(bestRes.get(0));
					recalls.add(bestRes.get(1));
					fMeasures.add(bestFmeasure);
					resultCSV+=cqa+","+bestRes.get(0)+","+bestFmeasure+","+bestRes.get(1)+"\n";

				}
			}



			resultCSV+="global mean,"+mean(precisions)+","+mean(fMeasures)+","+mean(recalls)+"\n";


			double nbEquivCQA = 0;
			double nbMoreGeneralCQA = 0;
			double nbMoreSpecificCQA = 0;
			double nbOverlapCQA = 0;
			for(int i =0; i <precisions.size();i++) {
				ArrayList<Double>result = new ArrayList<Double>();
				result.add(precisions.get(i));
				result.add(recalls.get(i));
				if(identical(result)) {
					nbEquivCQA++;
				}
				else if (QueryEvaluator.sourceMoreSpecificThanTarget(result)) {
					nbMoreGeneralCQA++;
				}
				else if (QueryEvaluator.sourceMoreGeneralThanTarget(result)) {
					nbMoreSpecificCQA++;
				}
				else if (overlap(result)) {
					nbOverlapCQA++;
				}
			}
			String proportionQueries="CQAs,"+nbEquivCQA/precisions.size()+","+(nbEquivCQA+nbMoreGeneralCQA+0.5*nbMoreSpecificCQA)/precisions.size()+
					","+(nbMoreSpecificCQA+nbEquivCQA+0.5*nbMoreGeneralCQA)/precisions.size()+","+
					(nbEquivCQA+nbOverlapCQA+nbMoreGeneralCQA+nbMoreSpecificCQA)/precisions.size()+","
					+ mean(fMeasures)+"\n";

			try {
				File theDir = new File(resultfile+"/CQA_coverage");

				// if the directory does not exist, create it
				if (!theDir.exists()) {
					System.out.println("creating directory: " + theDir.getName());
					try{
						theDir.mkdir();
					} 
					catch(SecurityException se){
						se.printStackTrace();
					}        
				}
				PrintWriter writer = new PrintWriter(resultfile+"/CQA_coverage/"+sourceOnto+"-"+targetOnto+".csv", "UTF-8");
				writer.println(resultCSV);
				writer.close(); 
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				e.printStackTrace();
			}
			try {
				PrintWriter writer = new PrintWriter(resultfile+"/"+sourceOnto+"-"+targetOnto+".csv", "UTF-8");
				writer.println(proportionQueries);
				writer.close(); 
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				e.printStackTrace();
			}


		} catch (Exception ex) {
			ex.printStackTrace();
		}


	}

	public static ArrayList<Double> compareHashSet(HashSet<String> hsource, HashSet<String> htarget) {
		ArrayList<Double> results= new ArrayList<Double>();
		int correctResults = 0;
		for (String targRes: htarget) {
			if (hsource.contains(targRes)) {
				correctResults ++;
			}
		}
		double prec = 0;
		double rec = 0;
		if(hsource.size()>0 && htarget.size()>0) {
			prec = (double)correctResults/(double)htarget.size();
			rec = (double)correctResults/(double)hsource.size();
		}
		results.add(prec);
		results.add(rec);
		return results;
	}

	public static double mean(ArrayList<Double> list) {
		double res = 0;
		for(double d: list) {
			res+=d;
		}		
		return res/(double)list.size();
	}

	public static boolean identical(ArrayList<Double> results) {
		return results.get(0)==1 && results.get(1) == 1;
	}

	public static boolean subsumption(ArrayList<Double> results) {
		return (results.get(0)==1 && results.get(1) > 0) ||
				(results.get(0)> 0 && results.get(1) ==1 ) ;
	}

	public static boolean overlap(ArrayList<Double> results) {
		return results.get(0)>0 && results.get(1) > 0;
	}

	public static boolean disjoint(ArrayList<Double> results) {
		return results.get(0)==0 && results.get(1) == 0;
	}

	public static double fMeasure(ArrayList<Double> results) {
		double prec = results.get(0);
		double rec = results.get(1);
		if(prec==0 && rec== 0) {
			return 0;
		}
		else {
			return 2*(prec*rec)/(prec+rec);
		}		
	}

	public static HashSet<String> getSPARQLQueryResults(String onto, String query){
		HashSet<String> results = new HashSet<String>();
		try {

			SparqlProxy spIn = SparqlProxy.getSparqlProxy("http://localhost:3030/"+onto+"_100");
			int offset = 0;
			int limit = 10000;
			boolean end = false;
			while (!end) {
				String newQuery = query;
				newQuery += "\n LIMIT "+limit;
				newQuery += "\n OFFSET "+offset;
				ArrayList<JsonNode> ret = spIn.getResponse(newQuery);
				Iterator<JsonNode> retIterator = ret.iterator();
				int nbAns = 0;
				while (retIterator.hasNext()) {
					nbAns++;
					JsonNode ans= retIterator.next();
					if(ans.has("s")) {
						String s =instanceString(ans.get("s").get("value").toString());
						String o = "";
						if(ans.has("o")) {
							o=instanceString(ans.get("o").get("value").toString());
						}
						results.add(s+o);
					}
				}
				if(nbAns <limit) {
					end = true;					
				}
				else {
					offset+=limit;
				}
				if (offset > 60000) {
					end = true;
				}
			}


		} catch (Exception e) {
			e.printStackTrace();
		}
		return results;
	}

	public static String getQueryContent(File f) {
		String result="";
		try {
			Scanner sc = new Scanner(f);

			while(sc.hasNextLine()) {
				String line = sc.nextLine();
				result+=line+" ";
			}
			sc.close();
		}
		catch(FileNotFoundException e) {
			e.printStackTrace();
		};
		return result;
	}


	public static String instanceString(String raw) {
		String result = raw.replaceAll("\"", "").replaceAll("http://[^\\#]+\\#","").replaceAll("_v0", "").replaceAll("_v2", "");
		return result;
	}

	public static String unprefixQuery(String query) {
		query=query.replaceAll("prefix", "PREFIX");
		Pattern pattern0 = Pattern.compile("PREFIX(.+):[ ]*<([^>]+)>");
		Matcher matcher0 = pattern0.matcher(query);
		HashMap<String,String> prefixes = new HashMap<String,String>();
		while  (matcher0.find()){ 
			//System.out.println(matcher0.group());
			prefixes.put(matcher0.group(1).trim(),matcher0.group(2));
		}
		query=query.replaceAll("PREFIX(.+):[ ]*<([^>]+)>", "");
		for(String key :prefixes.keySet()) {
			query = query.replaceAll(key+":([A-Za-z0-9_-]+)", "<"+prefixes.get(key)+"$1>");
		}
		return query;
	}
}
