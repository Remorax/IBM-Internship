package fr.irit.verificator;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Scanner;

import com.fasterxml.jackson.databind.JsonNode;

import fr.irit.melodi.sparql.proxy.SparqlProxy;


public class QueryVerificator {

	public static void main(String[] args) {
		File queriesFolder = new File("queries");
		File[] listOfOnto = queriesFolder.listFiles();
		for (int i = 0; i < listOfOnto.length; i++) {
			String targetCSV = "query,prec,rec,size_source,size_target,common\n";
			String sourceQueries="";
			File qListFile = new File(listOfOnto[i].getPath());
			File[] listOfQueries = qListFile.listFiles();
			String onto = listOfOnto[i].getName();
			System.out.println(onto);
			//for each query
			for (int j = 0; j < listOfQueries.length; j++) {
				//System.out.println(listOfQueries[j].isFile()+"--"+listOfQueries[j].getName());
				if(listOfQueries[j].isFile()) {					
					// Get results from query
					try {
						String queryContent = getQueryContent(listOfQueries[j]);
						HashSet<String> sourceResults = new HashSet<String>();
						SparqlProxy spIn = SparqlProxy.getSparqlProxy("http://localhost:3030/"+onto+"_100");
						ArrayList<JsonNode> ret = spIn.getResponse(queryContent);
						Iterator<JsonNode> retIterator = ret.iterator();
						while (retIterator.hasNext()) {
							JsonNode ans= retIterator.next();
							String s =instanceString(ans.get("s").get("value").toString());
							String o = "";
							if(ans.has("o")) {
								o=instanceString(ans.get("o").get("value").toString());
							}
							sourceResults.add(s+o);
						}
						sourceQueries+=listOfQueries[j].getName()+","+queryContent.replaceAll("SELECT distinct ", "").replaceAll("WHERE", "")
								.replaceAll("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>", "");

						File qTargetList = new File(listOfQueries[j].getPath().replaceAll(".sparql", ""));
						File[] listOfTargetQueries = qTargetList.listFiles();
						for (int k = 0; k < listOfTargetQueries.length; k++) {
							String queryTargetContent = getQueryContent(listOfTargetQueries[k]);
							HashSet<String> targetResults = new HashSet<String>();
							String targetOnto = listOfTargetQueries[k].getName().replaceAll("_[0-9]+_[0-9]+.sparql", "");
							SparqlProxy spTarg = SparqlProxy.getSparqlProxy("http://localhost:3030/"+targetOnto+"_100");
							ArrayList<JsonNode> retT = spTarg.getResponse(queryTargetContent);
							Iterator<JsonNode> retTIterator = retT.iterator();
							while (retTIterator.hasNext()) {
								JsonNode ans= retTIterator.next();
								String s = instanceString(ans.get("s").get("value").toString());
								String o = "";
								if(ans.has("o")) {
									o=instanceString(ans.get("o").get("value").toString());
								}
								targetResults.add(s+o);
							}
							sourceQueries+=","+targetOnto;

							//Calcul precision recall
							int correctResults = 0;
							for (String targRes: targetResults) {
								if (sourceResults.contains(targRes)) {
									correctResults ++;
								}
							}
							double prec = 0;
							double rec = 0;
							if(sourceResults.size()>0 && targetResults.size()>0) {
								 prec = (double)correctResults/(double)targetResults.size();
								 rec = (double)correctResults/(double)sourceResults.size();
							}
							
							targetCSV+=listOfTargetQueries[k].getName()+","+prec+","+rec+","+sourceResults.size()+","+targetResults.size()+","+correctResults+"\n";

						}
						sourceQueries+="\n";


					} catch (Exception e) {
						e.printStackTrace();
					}
				}//end for target query	
			}//end for source query

			try {
				PrintWriter writer = new PrintWriter("verif/"+onto+".csv", "UTF-8");
				writer.println(targetCSV);
				writer.close(); 
				PrintWriter writersou = new PrintWriter("verif/source_"+onto+".csv", "UTF-8");
				writersou.println(sourceQueries);
				writersou.close();
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				e.printStackTrace();
			}
		}//end for ontology

	}

	public static String getQueryContent(File f) {
		String result="";
		try {
			//System.out.println("Scanning "+f.getName());
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

}
