package fr.irit.population;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.lang3.StringUtils;
import org.json.simple.JSONObject;

public class Person {
	private String firstName;
	private ArrayList<String> notSure;
	private String lastName;
	private String name;
	HashSet<String> otherNames;
	HashSet<String> differentPeopleNames;
	private boolean attendee;
	private boolean student;
	private boolean earlyRegistration;
	public String IRI;
	private boolean isPCMember;
	private boolean isOCMember;
	private boolean isSCMember;
	private boolean forcedNotStudent;
	private Organisation organisation;

	public Person(String dbpediaIRI, String name, Organisation org) {
		this.name=name;
		Pattern pattern = Pattern.compile("http://dbpedia.org/resource/([A-Z_a-z]+)_([A-Za-z]+)");
		Matcher matcher = pattern.matcher(dbpediaIRI);
		while (matcher.find()){
			this.firstName=matcher.group(1).trim();
			this.lastName=matcher.group(2).trim();
		}
		this.organisation=org;
		this.isOCMember=false;
		this.forcedNotStudent = false;
		this.isPCMember=false;
		this.isSCMember=false;
	//	System.out.println("Person: "+this.firstName+" "+this.lastName);

	}

	public Person(String firstName, String lastName, String name, String prefix) {
		this.firstName=firstName;
		this.lastName=lastName;
		this.name = name;
		this.IRI = this.personIRI(prefix);
		otherNames=new HashSet<String>();
		otherNames.add(name.toLowerCase());
		differentPeopleNames = new HashSet<String>();
	}

	public Person(String name, String prefix) {
		earlyRegistration = false;
		isPCMember = false;
		differentPeopleNames = new HashSet<String>();
		otherNames=new HashSet<String>();
		otherNames.add(name.toLowerCase());
		Pattern pattern = Pattern.compile("([^ ]+) *");
		Matcher matcher = pattern.matcher(name);
		ArrayList<String> names = new ArrayList<String>();
		notSure = new ArrayList<String>();
		while (matcher.find()){
			names.add(matcher.group().trim());
		}
		if (names.size()==2) {
			firstName = names.get(0);
			lastName = names.get(1);
		}
		else if (names.size() > 2) {
			firstName = names.get(0);
			lastName = names.get(names.size()-1);
			for (int i = names.size()-2 ; i >0; i --) {
				if(names.get(i).toLowerCase().equals("de") ||
						names.get(i).toLowerCase().equals("van") ||
						names.get(i).toLowerCase().equals("den")||
						names.get(i).toLowerCase().equals("do")||
						names.get(i).toLowerCase().equals("di")) {
					this.lastName = names.get(i)+ " "+this.lastName;
				}
				else if (!names.get(i).contains(".") || names.size()>1){
					notSure.add(names.get(i));
				}
			}
		}
		this.name  =name;
		this.IRI = personIRI(prefix);
	}

	public void attendsConference() {
		this.attendee = true;		
	}

	public void forceEarlyRegistration() {
		this.earlyRegistration = true;
	}


	public String getName() {
		return this.name;
	}
	public String getFirstName() {
		return this.firstName;
	}
	public String getLastName() {
		return this.lastName;
	}

	public boolean isAttendee() {
		return this.attendee;
	}
	public boolean isEarlyRegistration() {
		return this.earlyRegistration;
	}


	//Person p is newbie so it has only 1 name
	public double differenceWith(Person p) {
		double difference= 1.0;
		String s1= p.name.toLowerCase();
		boolean stopSearching = false;
		// same name : no difference
		if (otherNames.contains(s1)) {
			difference = 0.0;
			stopSearching =true;
		} 
		else if (differentPeopleNames.contains(s1)) {
			difference=1.0;
			stopSearching = true;
		}
		//one of the names included in the other
		else {
			Iterator <String> it  = this.otherNames.iterator();
			while(it.hasNext() && !stopSearching) {
				String myName = it.next();
				if (s1.contains(myName) || myName.contains(s1)) {
					difference = 0.0;
					stopSearching = true;
				}
				else {
					difference = StringUtils.getLevenshteinDistance(s1.toLowerCase(), myName.toLowerCase())/((double)Math.max(s1.length(), myName.length()));
				}
			}
		}		
		return difference;
	}

	public String personIRI(String prefix) {
		return "<"+prefix+"person"+this.name.hashCode()+">";
	}

	public String toString() {
		//return otherNames.toString();
		return "firstname: "+this.firstName+" - lastname: "+this.lastName+ " - notsure: "+this.notSure;
	}
	public boolean hasNotSureField() {
		return !notSure.isEmpty();
	}
	public void attachNotSureToFirstName(String str) {
		firstName = firstName + " "+str;
	}
	public void attachNotSureToLasName(String str) {
		lastName = str + " "+lastName;
	}
	public void addNewName(String n2) {
		otherNames.add(n2.toLowerCase());
	}
	public void differentPerson(Person p) {
		for(String pName: p.otherNames) {
			differentPeopleNames.add(pName);
		}
	}
	public void forcePCMember() {
		this.isPCMember = true;
	}
	public void forceSCMember() {
		this.isSCMember = true;
	}
	public void forceOCMember() {
		this.isOCMember = true;
	}
	public void forceStudent() {
		this.organisation.forceUniversity();
		this.student = true;
	}
	public void forceNotStudent() {
		this.student = false;
		this.forcedNotStudent=true;
	}
	public boolean isStudent() {
		return this.student;
	}

	public boolean isPCMember() {
		return this.isPCMember;
	}
	public boolean isAMember() {
		return this.isOCMember || this.isPCMember || this.isSCMember;
	}
	public ArrayList<String> getNotSureStrings(){
		return this.notSure;
	}

	public Organisation getOrganisation() {
		return this.organisation;
	}
	public void setOrganisation(Organisation org) {
		this.organisation=org;
	}

	public void printCSV() {
		System.out.println(this.name+","+this.firstName+","+this.lastName+","+this.attendee+","+this.earlyRegistration);
	}

	public JSONObject toJSONObject() {
		JSONObject me = new JSONObject();
		me.put("first_name", this.firstName);
		me.put("last_name", this.lastName);
		me.put("name", this.name);
		me.put("student", ""+this.student);
		me.put("organisation", this.organisation.getName());
		me.put("participant", ""+this.attendee);
		me.put("early-registration", ""+this.attendee);
		me.put("organisation_type", this.organisation.getType());
		return me;
	}

	public void randomSelectStudent() {
		Random rand = new Random();
		if (!this.isAMember() || forcedNotStudent) {
			double val = rand.nextDouble();
			if (val <0.4) {
				this.forceStudent();
			}
			else {
				this.student=false;
			}

		}
	}

	public void randomSelectParticipant() {
		Random rand = new Random();
		double val = rand.nextDouble();
		if (!attendee) {
			if (val <0.4) {
				this.attendee = true;
				double val2 = rand.nextDouble();
				if(val2 < 0.45) {
					this.earlyRegistration=false;
				}
				else {
					this.earlyRegistration=true;
				}
			}
			else {
				this.attendee=false;
			}
		}
		else {
			double val2 = rand.nextDouble();
			if(val2 < 0.45) {
				this.earlyRegistration=false;
			}
			else {
				this.earlyRegistration=true;
			}
		}

	}


}
