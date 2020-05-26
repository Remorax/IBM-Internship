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

import com.fasterxml.jackson.databind.JsonNode;

import fr.irit.melodi.sparql.proxy.SparqlProxy;

public class QueryEvaluator {

	public static void main(String[] args) {

		File outputQueryFolder = new File(args[0]);
		String resultfile = "results/";
		if(args.length>1) {
			resultfile=args[1];
		}
		String sourceOnto ="";
		String targetOnto ="";
		Pattern pattern1 = Pattern.compile("([^-]+)-([^-]+)");
		Matcher matcher1 = pattern1.matcher(outputQueryFolder.getName());
		if (matcher1.find()){
			sourceOnto = matcher1.group(1);
			targetOnto = matcher1.group(2);
		}	

		//System.out.println(sourceOnto);
		//System.out.println(targetOnto);

		String resultCQACoverageCSV = "cqa,best_q_prec,best_q_fmeasure,best_q_rec\n";
		String resultOutputPrecisionCSV = "query,prec,fmeasure,rec\n";

		//ArrayList Results
		ArrayList<Double> precisions = new ArrayList<Double>();
		ArrayList<Double> recalls = new ArrayList<Double>();
		ArrayList<Double> fmeasures = new ArrayList<Double>();
		ArrayList<Double> bestPrecisions = new ArrayList<Double>();
		ArrayList<Double> bestRecalls = new ArrayList<Double>();
		ArrayList<Double> bestFmeasures = new ArrayList<Double>();

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
				String targetCQA = getQueryContent(targetCQAFile);
				HashSet<String> targetCQAresults = getSPARQLQueryResults(targetOnto,targetCQA);

				ArrayList<Double> bestRes = new ArrayList<Double>();
				bestRes.add(0.0);
				bestRes.add(0.0);
				double bestFmeasure = 0.0;
				File outputQueryFolderForCQA = new File(outputQueryFolder+"/"+cqa);
				if(outputQueryFolderForCQA.exists()) {
					File[] listOfOutputQueries = outputQueryFolderForCQA.listFiles();

					for(File queryFile: listOfOutputQueries) {
						//System.out.println(queryFile.getName());
						String query = getQueryContent(queryFile);
						HashSet<String> rewrittenResults = getSPARQLQueryResults(targetOnto,query);
						ArrayList<Double> rewrittenRes = compareHashSet(targetCQAresults,rewrittenResults);
						if (fMeasure(rewrittenRes) > bestFmeasure) {
							bestRes = rewrittenRes;
							bestFmeasure=fMeasure(rewrittenRes);
						}
						precisions.add(rewrittenRes.get(0));
						recalls.add(rewrittenRes.get(1));
						fmeasures.add(fMeasure(rewrittenRes));
						resultOutputPrecisionCSV+=queryFile.getName()+","+rewrittenRes.get(0)+","+fMeasure(rewrittenRes)+","+rewrittenRes.get(1)+"\n";
					}
				}
				bestPrecisions.add(bestRes.get(0));
				bestRecalls.add(bestRes.get(1));
				bestFmeasures.add(bestFmeasure);
				resultCQACoverageCSV+=cqa+","+bestRes.get(0)+","+bestFmeasure+","+bestRes.get(1)+"\n";
			}
			
			
			// If only sourceCQA exists and outputQueryFolderForCQA exists,
			else if (sourceCQAFile.exists() && !targetCQAFile.exists()) {
				// compare source results and outputQueryFolderForCQA results
				File outputQueryFolderForCQA = new File(outputQueryFolder+"/"+cqa);
				if(outputQueryFolderForCQA.exists()) {
					File[] listOfOutputQueries = outputQueryFolderForCQA.listFiles();
					String sourceCQA = getQueryContent(sourceCQAFile);
					HashSet<String> sourceCQAresults = getSPARQLQueryResults(sourceOnto,sourceCQA);

					for(File queryFile: listOfOutputQueries) {
						//System.out.println(queryFile.getName());
						String query = getQueryContent(queryFile);
						HashSet<String> rewrittenResults = getSPARQLQueryResults(targetOnto,query);
						ArrayList<Double> rewrittenRes = compareHashSet(sourceCQAresults,rewrittenResults);
						precisions.add(rewrittenRes.get(0));
						recalls.add(rewrittenRes.get(1));
						fmeasures.add(fMeasure(rewrittenRes));
						resultOutputPrecisionCSV+=queryFile.getName()+","+rewrittenRes.get(0)+","+fMeasure(rewrittenRes)+","+rewrittenRes.get(1)+"\n";
					}
				}
				
			}
		}
		
		//for each result: give the number of output queries which are equivalent, more general, more specific, overlap and disjoint from the CQA
		double nbEquiv = 0;
		double nbMoreGeneral = 0;
		double nbMoreSpecific = 0;
		double nbOverlap = 0;
		for(int i =0; i <precisions.size();i++) {
			ArrayList<Double>result = new ArrayList<Double>();
			result.add(precisions.get(i));
			result.add(recalls.get(i));
			if(identical(result)) {
				nbEquiv++;
			}
			else if (sourceMoreSpecificThanTarget(result)) {
				nbMoreGeneral++;
			}
			else if (sourceMoreGeneralThanTarget(result)) {
				nbMoreSpecific++;
			}
			else if (overlap(result)) {
				nbOverlap++;
			}
		}
		String proportionQueries = "type,equiv,recall oriented, precision oriented, overlap, f-measure\n";
		proportionQueries+="queries,"+nbEquiv/precisions.size()+","+(nbEquiv+nbMoreGeneral+0.5*nbMoreSpecific)/precisions.size()+
				","+(nbMoreSpecific+nbEquiv+0.5*nbMoreGeneral)/precisions.size()+","+
				(nbEquiv+nbOverlap+nbMoreGeneral+nbMoreSpecific)/precisions.size()+","
						+ mean(fmeasures)+"\n";
		//TODO: same with CQA
		
		double nbEquivCQA = 0;
		double nbMoreGeneralCQA = 0;
		double nbMoreSpecificCQA = 0;
		double nbOverlapCQA = 0;
		for(int i =0; i <bestPrecisions.size();i++) {
			ArrayList<Double>result = new ArrayList<Double>();
			result.add(bestPrecisions.get(i));
			result.add(bestRecalls.get(i));
			if(identical(result)) {
				nbEquivCQA++;
			}
			else if (sourceMoreSpecificThanTarget(result)) {
				nbMoreGeneralCQA++;
			}
			else if (sourceMoreGeneralThanTarget(result)) {
				nbMoreSpecificCQA++;
			}
			else if (overlap(result)) {
				nbOverlapCQA++;
			}
		}
		proportionQueries+="CQAs,"+nbEquivCQA/bestPrecisions.size()+","+(nbEquivCQA+nbMoreGeneralCQA+0.5*nbMoreSpecificCQA)/bestPrecisions.size()+
				","+(nbMoreSpecificCQA+nbEquivCQA+0.5*nbMoreGeneralCQA)/bestPrecisions.size()+","+
				(nbEquivCQA+nbOverlapCQA+nbMoreGeneralCQA+nbMoreSpecificCQA)/bestPrecisions.size()+","
						+ mean(bestFmeasures)+"\n";


		resultOutputPrecisionCSV+="global mean,"+mean(precisions)+","+mean(fmeasures)+","+mean(recalls)+"\n";
		resultCQACoverageCSV+="global mean,"+mean(bestPrecisions)+","+mean(bestFmeasures)+","+mean(bestRecalls)+"\n";
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
			writer.println(resultCQACoverageCSV);
			writer.close(); 
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}


		try {
			File theDir = new File(resultfile+"/Precision");

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
			PrintWriter writer = new PrintWriter(resultfile+"/Precision/"+sourceOnto+"-"+targetOnto+".csv", "UTF-8");
			writer.println(resultOutputPrecisionCSV);
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

	public static boolean sourceMoreGeneralThanTarget(ArrayList<Double> results) {
		return (results.get(0)==1 && results.get(1) > 0) ;
	}
	
	public static boolean sourceMoreSpecificThanTarget(ArrayList<Double> results) {
		return 	(results.get(0)> 0 && results.get(1) ==1 ) ;
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
		if(prec==0 && rec == 0) {
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
