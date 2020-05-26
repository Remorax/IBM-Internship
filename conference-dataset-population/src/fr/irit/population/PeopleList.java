package fr.irit.population;

import java.util.Collection;
import java.util.HashMap;
import java.util.Scanner;


public class PeopleList {
	
	private HashMap<String,Person> people;
	private HashMap<String,String> peopleNameIRI;
	//Scanner reader ;
	
	public PeopleList() {
		people = new HashMap<String,Person>();
		peopleNameIRI = new HashMap<String,String>();
		//reader = readerIn;
	}
	public String addPerson(Person p) {
		people.put(p.IRI, p);
		peopleNameIRI.put(p.getName(),p.IRI);
		return p.IRI;
	}
	
	public String addPerson(String name,String prefix, Scanner reader) {
		
		//Person newbie = new Person(name,prefix);
		//System.out.println(newbie.toString());
		
		/*Iterator<Person> it = people.values().iterator();
		boolean foundSame = false;
		while(it.hasNext() && !foundSame) {
			Person p = it.next();
			if (p.differenceWith(newbie)==0.0) {
				//Same individual found - no addition to people - return existing IRI
				foundSame=true;
				IRI = p.IRI;
			}
			else if (p.differenceWith(newbie)<=0.25) {
				//ask user about same individual or not	
				System.out.println("New Person : "+name+ " - Existing Person: "+p);
				System.out.println("Press:\n"
						+ "1- if they are the same person\n"
						+ "2- if they are two different people");
				
				String n = reader.next(); 
				//String n="1";
				//keep if same
				if (n.equals("1")) {
					//System.out.println("They are indeed the same");
					foundSame=true;
					IRI=p.IRI;
					p.addNewName(name);
				}
				else if (n.equals("2")) {
					p.differentPerson(newbie);
					newbie.differentPerson(p);
				}
			}	
		}
		// if the new person doesnot exist, add it to people
		if(!foundSame) {
			people.put(newbie.IRI,newbie);
		}
		return IRI;*/
		
		return peopleNameIRI.get(name);
		
	}
	
	public Person getPersonFromIRI(String IRI) {
		return people.get(IRI);		
	}
		
	public Collection<Person> getPeople(){
		return people.values();
	}

	public void printContent() {
		for (Person p : people.values()) {
			p.printCSV();
		}
	}
	

}
