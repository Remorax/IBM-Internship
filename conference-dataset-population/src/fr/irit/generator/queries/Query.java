package fr.irit.generator.queries;

import java.util.HashSet;

public class Query {

	private String ontology;
	private String graphPattern;
	private String type;
	private HashSet<Query> matchedQueries;

	public Query(String GP, String onto) {
		graphPattern=GP;
		ontology=onto;
		if(GP.contains("?o ")) {
			type = "property";
		}
		else {
			type = "class";
		}
		matchedQueries=new HashSet<Query>();
	}

	public void addMatchedQuery(Query q) {
		this.matchedQueries.add(q);
	}

	public boolean isClass() {
		return this.type.equals("class");
	}
	public boolean isProperty() {
		return this.type.equals("property");
	}

	public boolean isSimpleClass() {
		if(this.isProperty()) {
			return false;
		}
		else {
			return this.graphPattern.matches("\\?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> [^\\.]+ .[^\\.]*");
		}
	}

	public boolean isSimpleProperty() {
		if(this.isClass()) {
			return false;
		}
		else {
			return this.graphPattern.matches("\\?s [^\\?]+ \\?o .[^\\.]*");
		}
	}

	public int hashCode(){
		int hashcode = this.toSPARQLQuery().hashCode();
		return hashcode;
	}

	public boolean equals(Object obj){
		if (obj instanceof Query) {
			Query pp = (Query) obj;
			return (pp.toSPARQLQuery().equals(this.toSPARQLQuery()));
		} else {
			return false;
		}
	}

	public String getOntology() {
		return this.ontology;
	}
	
	public HashSet<Query> getTargetQueries() {
		return this.matchedQueries;
	}

	public String toSPARQLQuery() {
		String vars ="?s";
		if(this.isProperty()) {
			vars+=" ?o";
		}
		String query = "SELECT distinct "+vars+ " WHERE {  \n"
				+ this.graphPattern + "  \n }";
		return query;
	}
}
