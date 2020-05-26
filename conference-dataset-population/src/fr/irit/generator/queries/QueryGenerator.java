package fr.irit.generator.queries;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class QueryGenerator {

	public static void main(String[] args) {
		// liste des requetes
		// pour chaque requete source, une liste de requetes associées

		//for each sparql construct alignment
		File folder = new File("alignments/construct");
		File[] listOfFiles = folder.listFiles();
		HashMap<String,HashMap<String,Query>> queriesPerOnto = new HashMap<String, HashMap<String,Query>>();

		for (int i = 0; i < listOfFiles.length; i++) {
			String fileName = listOfFiles[i].getName();
			Pattern pattern = Pattern.compile("([^-]+)-([^-]+).sparql");
			Matcher matcher = pattern.matcher(fileName);
			String o1 = "";
			String o2 = "";
			if (matcher.find()){ 
				o1 = matcher.group(1).trim();
				o2 = matcher.group(2).trim();
			}
			if(!queriesPerOnto.containsKey(o1)) {
				queriesPerOnto.put(o1,new HashMap<String,Query>());
			}
			if(!queriesPerOnto.containsKey(o2)) {
				queriesPerOnto.put(o2,new HashMap<String,Query>());
			}
			String content = "";
			try {
				Scanner sc = new Scanner(listOfFiles[i]);

				while(sc.hasNextLine()) {
					String line = sc.nextLine();
					if(line.contains("PREFIX")) {
						content+="END\n"+line+"\n";
					}
					content+=line+" ";


				}
				content+=" END";
				sc.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			};

			Pattern pattern0 = Pattern.compile("PREFIX ([^:]+):<([^>]+)>");
			Matcher matcher0 = pattern0.matcher(content);
			HashMap<String,String> prefixes = new HashMap<String,String>();
			while  (matcher0.find()){ 
				prefixes.put(matcher0.group(1),matcher0.group(2));
			}
			content=content.replaceAll("PREFIX [^:]+:<[^>]+>", "");
			for(String key :prefixes.keySet()) {
				content = content.replaceAll(key+":([^ ]+)", "<"+prefixes.get(key)+"$1>");
			}

			//System.out.println(content);
			//System.out.println(prefixes);

			Pattern pattern1 = Pattern.compile("CONSTRUCT \\{ (.+)\\} WHERE \\{ (.+)\\}[ ]*END");
			Matcher matcher1 = pattern1.matcher(content);
			int nfound = 0;
			while  (matcher1.find()){ 
				nfound++;
				//System.out.println(matcher1.group());
				// get construct GP1 GP2
				String targetGP = matcher1.group(1).trim();
				String sourceGP = matcher1.group(2).trim();

				Query qsource = new Query(sourceGP,o1);
				Query qtarget = new Query(targetGP,o2);

				if (queriesPerOnto.get(o1).containsKey(qsource.toSPARQLQuery())) {
					qsource = queriesPerOnto.get(o1).get(qsource.toSPARQLQuery());
				}
				if (queriesPerOnto.get(o2).containsKey(qtarget.toSPARQLQuery())) {
					qtarget = queriesPerOnto.get(o2).get(qtarget.toSPARQLQuery());
				}
				// if GP1 simple class & GP2 simple class -- idem simple properties
				boolean added=false;
				if(qsource.isSimpleClass()) {
					qsource.addMatchedQuery(qtarget);
					added=true;

					if (!queriesPerOnto.get(o1).containsKey(qsource.toSPARQLQuery())) {
						queriesPerOnto.get(o1).put(qsource.toSPARQLQuery(),qsource);
					}
				}
				if(qtarget.isSimpleClass()) {
					qtarget.addMatchedQuery(qsource);
					added=true;
					if (!queriesPerOnto.get(o2).containsKey(qtarget.toSPARQLQuery())) {
						queriesPerOnto.get(o2).put(qtarget.toSPARQLQuery(),qtarget);
					}
				}
				if(qsource.isSimpleProperty()) {
					qsource.addMatchedQuery(qtarget);
					added=true;
					if (!queriesPerOnto.get(o1).containsKey(qsource.toSPARQLQuery())) {
						queriesPerOnto.get(o1).put(qsource.toSPARQLQuery(),qsource);
					}
				}
				if(qtarget.isSimpleProperty()) {
					qtarget.addMatchedQuery(qsource);
					added=true;
					if (!queriesPerOnto.get(o2).containsKey(qtarget.toSPARQLQuery())) {
						queriesPerOnto.get(o2).put(qtarget.toSPARQLQuery(),qtarget);
					}
				}
				if(!added) {
					System.out.println(qsource.toSPARQLQuery()+"-"+qtarget.toSPARQLQuery());
				}
			}
		//	System.out.println(o1+"--"+o2+"  :  "+nfound);

		}

		// For each ontology, create source query,
		for(String onto: queriesPerOnto.keySet()) {
			//for each source query I, create set of target queries name onto_I_num
			try {
				int i = 0;
				System.out.println(onto+"--"+queriesPerOnto.get(onto).size());
				for(Query q : queriesPerOnto.get(onto).values()) {
					//System.out.println(q.getTargetQueries().size()+" -- "+q.toSPARQLQuery());
					File theDir = new File("queries/"+onto);

					// if the directory does not exist, create it
					if (!theDir.exists()) {
						//  System.out.println("creating directory: " + theDir.getName());
						boolean result = false;

						try{
							theDir.mkdir();
							result = true;
						} 
						catch(SecurityException se){
							//handle it
						} 
					}
					PrintWriter writers = new PrintWriter("queries/"+onto+"/q_"+i+".sparql", "UTF-8");
					writers.println(q.toSPARQLQuery());
					writers.close();
					int j =0;
					for(Query qt: q.getTargetQueries()) {
						File theDir2 = new File("queries/"+onto+"/q_"+i);

						// if the directory does not exist, create it
						if (!theDir2.exists()) {
							//    System.out.println("creating directory: " + theDir2.getName());
							boolean result = false;

							try{
								theDir2.mkdir();
								result = true;
							} 
							catch(SecurityException se){
								//handle it
							} 
						}
						PrintWriter writer = new PrintWriter("queries/"+onto+"/q_"+i+"/"+qt.getOntology()+"_"+i+"_"+j+".sparql", "UTF-8");
						writer.println(qt.toSPARQLQuery());
						writer.close();

						j++;
					}
					i++;
				}
			} catch (FileNotFoundException |UnsupportedEncodingException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}




	}

}
