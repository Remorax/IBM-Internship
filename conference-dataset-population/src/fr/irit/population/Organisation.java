package fr.irit.population;

import java.util.Random;

public class Organisation {
	private String name;
	private String IRI;
	private String type;
	
	public Organisation(String name) {
		this.name = name;
		Random r = new Random();
		boolean isUniversity=(r.nextInt(10)>=3);
		if (isUniversity) {
			this.type="university";
		}
		else {
			this.type = "company";
		}
	}
	
	public Organisation(String name, String type,String prefix) {
		if (name.equals("")) {
			this.name="Poule University";
		}else {
			this.name=name;
		}
		this.type = type;
		this.IRI= orgaIRI(prefix);
	}

	public String orgaIRI(String prefix) {
		return "<"+prefix+"orga"+this.name.hashCode()+">";
	}
	public void forceUniversity() {
		this.type="university";
	}
	
	public boolean isUniversity() {
		return this.type.equals("university");
	}
	public boolean isCompany() {
		return this.type.equals("company");
	}
	public String getIRI() {
		return this.IRI;
	}
	public String getName() {
		return this.name;
	}
	public String getType() {
		return this.type;
	}
}
