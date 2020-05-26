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


public class CQAVerificator {

	public static void main(String[] args) {
		File queriesFolder = new File("CQAs");
		File[] listOfCQA = queriesFolder.listFiles();
		String targetCSV = "CQA,onto,size\n";
		
		for (int i = 0; i < listOfCQA.length; i++) {
			
			File qListFile = new File(listOfCQA[i].getPath());
			File[] listOfOnto = qListFile.listFiles();
			String cqa = listOfCQA[i].getName();
			System.out.println(cqa);
			//for each query
			for (int j = 0; j < listOfOnto.length; j++) {
				//System.out.println(listOfQueries[j].isFile()+"--"+listOfQueries[j].getName());
				if(listOfOnto[j].isFile()) {					
					// Get results from query
					try {
						String onto = listOfOnto[j].getName().replaceAll(".sparql", "");
						String queryContent = getQueryContent(listOfOnto[j]);
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
						targetCSV+=cqa+","+onto+","+sourceResults.size();
						targetCSV+="\n";


					} catch (Exception e) {
						e.printStackTrace();
					}
				}//end for target query	
			}//end for source query
			
		}//end for ontology
		try {
			PrintWriter writer = new PrintWriter("verif/cqa.csv", "UTF-8");
			writer.println(targetCSV);
			writer.close(); 
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}

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
