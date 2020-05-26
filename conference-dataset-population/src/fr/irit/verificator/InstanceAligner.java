package fr.irit.verificator;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

import com.fasterxml.jackson.databind.JsonNode;

import fr.irit.melodi.sparql.proxy.SparqlProxy;

public class InstanceAligner {

	public static void main(String[] args) {
		ArrayList<String> ontologies = new ArrayList<String>();
		ontologies.add("cmt");
		ontologies.add("conference");
		ontologies.add("confOf");
		ontologies.add("edas");
		ontologies.add("ekaw");
		HashMap<String,String> alignments = new HashMap<String,String>();
		HashMap<String,HashSet<String>> instances = new HashMap<String,HashSet<String>>();
		
		//set of equivalent iris
		HashMap<String,ArrayList<String>> originalIRIs = new HashMap<String,ArrayList<String>>();

		String instanceQuery = "SELECT distinct ?x where{ "
				+ "?x a ?y. ?y a <http://www.w3.org/2002/07/owl#Class>. "
				+ "FILTER(isIRI(?x)) }";

		for(String onto: ontologies) {
			alignments.put(onto, "");
			instances.put(onto, new HashSet<String>());
			try {
				SparqlProxy spIn = SparqlProxy.getSparqlProxy("http://localhost:3030/"+onto+"_80");
				ArrayList<JsonNode> ret = spIn.getResponse(instanceQuery);
				Iterator<JsonNode> retIterator = ret.iterator();
				while (retIterator.hasNext()) {
					JsonNode ans= retIterator.next();
					String iri = ans.get("x").get("value").toString().replaceAll("\"", "");
					instances.get(onto).add(iri);
					if(originalIRIs.containsKey(instanceString(iri))) {
						originalIRIs.get(instanceString(iri)).add(iri);
					}
					else {
						ArrayList<String> instanceIRI = new ArrayList<String>();
						instanceIRI.add(iri);
						originalIRIs.put(instanceString(iri),instanceIRI);
					}
					
				}

			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		for(String onto: ontologies) {
			System.out.println(onto);
			for(String iri: instances.get(onto)) {
				//if more than one iri in set of equivalent iris
				if(originalIRIs.get(instanceString(iri)).size()>1) {
					for(String targetIRI: originalIRIs.get(instanceString(iri))) {
						if(!targetIRI.contains("http://"+onto)) {
							alignments.put(onto, alignments.get(onto)
									+ " <"+iri+"> <http://www.w3.org/2002/07/owl#sameAs> <"+targetIRI+"> .\n");
						//	System.out.println(iri+" "+targetIRI);
						}
					}
				}				
				
			}
			try {
				PrintWriter writer = new PrintWriter("populated_datasets/data_80/alignment_"+onto+"_80.ttl", "UTF-8");
				writer.println(alignments.get(onto));
				writer.close(); 
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				e.printStackTrace();
			}
		}
		

	}

	public static String instanceString(String raw) {
		String result = raw.replaceAll("\"", "").replaceAll("http://[^\\#]+\\#","").replaceAll("_v0", "").replaceAll("_v2", "");
		return result;
	}

}
