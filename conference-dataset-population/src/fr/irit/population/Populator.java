package fr.irit.population;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import fr.irit.melodi.sparql.exceptions.IncompleteSubstitutionException;
import fr.irit.melodi.sparql.exceptions.NotAFolderException;
import fr.irit.melodi.sparql.files.FolderManager;
import fr.irit.melodi.sparql.proxy.SparqlProxy;
import fr.irit.melodi.sparql.query.exceptions.SparqlEndpointUnreachableException;
import fr.irit.melodi.sparql.query.exceptions.SparqlQueryMalFormedException;

public class Populator {
	private  static Scanner reader;
	private static void printUsage() {
		System.err.println("Usage : java -jar Populator.jar parameterfile.json");
	}

	public static void main(String[] args) {
		if (args.length < 1) {
			printUsage();
			System.exit(1);
		}
		JSONParser parser = new JSONParser();
		reader  = new Scanner(System.in);
		try {			
			JSONObject parameters = (JSONObject) (parser.parse(new FileReader(args[0])));
			JSONArray ontologyParameters = (JSONArray)parameters.get("ontologies");
			for(Object ontoObj: ontologyParameters) {
				JSONObject onto = (JSONObject)ontoObj;
				String ontoName = onto.get("name").toString();
				String instPrefix = onto.get("instance_prefix").toString();
				String endpoint = onto.get("endpoint").toString();
				SparqlProxy spEndpoint = SparqlProxy.getSparqlProxy(endpoint);
				Map<String,FolderManager> queryTemplates = new HashMap<>();
				queryTemplates.put(ontoName, new FolderManager("query_templates/"+ontoName+"/"));
				queryTemplates.get(ontoName).loadQueries();
				System.out.println("Ontology : " + ontoName);
				JSONArray confFileNames = (JSONArray)  onto.get("conference_file_path");
				for(Object confFileObj: confFileNames) {
					System.out.println(confFileObj.toString());
					readConferenceFile(ontoName,confFileObj.toString(),instPrefix,queryTemplates,spEndpoint);
				}				
				System.out.println("inference");
				basicInference(spEndpoint, queryTemplates);
			}

		} catch (IOException | ParseException |  NotAFolderException | SparqlQueryMalFormedException | SparqlEndpointUnreachableException e) {
			e.printStackTrace();
		}
		//reader.close();




	}


	public static void readConferenceFile(String ontologyname, String conferenceFilePath, 
			String prefix,Map<String,FolderManager> queryTemplates,SparqlProxy spEndpoint) {

		JSONParser parser = new JSONParser();
		PeopleList people = new PeopleList();
		Map<String, String> substitution = new HashMap<>();
		HashMap<String,Organisation> organisations = new HashMap<>();
		try {
			//CONFERENCE
			JSONObject conference = (JSONObject) (parser.parse(new FileReader(conferenceFilePath)));
			
			JSONArray persons = (JSONArray) (parser.parse(new FileReader(conference.get("people_file").toString())));
			for(Object personFromFileObj: persons) {
				JSONObject personFromFile = (JSONObject) personFromFileObj;
				Person p = new Person(personFromFile.get("first_name").toString(),personFromFile.get("last_name").toString(),personFromFile.get("name").toString(),prefix);
				people.addPerson(p);
				if(personFromFile.get("participant").toString().equals("true")) {
					p.attendsConference();
					if(personFromFile.get("early-registration").toString().equals("true")) {
						p.forceEarlyRegistration();
					}
				}
				Organisation org = new Organisation(personFromFile.get("organisation").toString(),
						personFromFile.get("organisation_type").toString(),prefix);
				organisations.put(org.getIRI(), org);
				p.setOrganisation(org);
				if(personFromFile.get("student").toString().equals("true")) {
					p.forceStudent();
				}
				
				
			}

			
			String confTitle = conference.get("title").toString();
			int confHash = confTitle.hashCode();
			String confIRI = makeIRI("conference"+confHash,prefix);
			substitution.put("conf",confIRI);
			substitution.put("conftitle", makeLiteral(conference.get("title").toString()));
			substitution.put("confnumrev", makeLiteral(conference.get("num_reviews").toString())+"^^xsd:int");
			substitution.put("confshorttitle", makeLiteral(conference.get("short_title").toString()));
			substitution.put("logoURL", makeLiteral(conference.get("logo_url").toString()));
			substitution.put("confURL", makeLiteral(conference.get("url").toString()));
			substitution.put("confwww", makeIRI("confWebSite"+confHash,prefix));
			substitution.put("confwwwname", makeLiteral("Website of "+confTitle));
			substitution.put("conflocinst", makeIRI("location"+conference.get("location").toString().hashCode(),prefix));
			substitution.put("conflocation", makeLiteral(conference.get("location").toString()));

			substitution.put("startDate", makeLiteral(conference.get("start").toString()));
			substitution.put("endDate", makeLiteral(conference.get("end").toString()));
			substitution.put("absDate", makeLiteral(conference.get("abstract_deadline").toString()));
			substitution.put("subDate", makeLiteral(conference.get("submission_deadline").toString()));
			substitution.put("notifDate", makeLiteral(conference.get("review_deadline").toString()));
			substitution.put("camDate", makeLiteral(conference.get("camera-ready_deadline").toString()));
			substitution.put("regisDate", makeLiteral(conference.get("registration_deadline").toString()));

			substitution.put("startDateInst", makeIRI("startDate"+confHash,prefix));
			substitution.put("endDateInst", makeIRI("endDate"+confHash,prefix));
			substitution.put("absDateInst", makeIRI("absDate"+confHash,prefix));
			substitution.put("subDateInst", makeIRI("subDate"+confHash,prefix));
			substitution.put("notifDateInst", makeIRI("notifDate"+confHash,prefix));
			substitution.put("camDateInst", makeIRI("camDateInst"+confHash,prefix));

			substitution.put("subEventInst", makeIRI("subEvt"+confHash,prefix));
			substitution.put("reviewEventInst", makeIRI("revEvt"+confHash,prefix));
			substitution.put("notifEventInst", makeIRI("notifEvt"+confHash,prefix));
			substitution.put("camEventInst", makeIRI("camEvt"+confHash,prefix));
			substitution.put("regisEventInst", makeIRI("regisEvt"+confHash,prefix));

			substitution.put("subCallname", makeLiteral("Call for papers for "+confTitle));
			substitution.put("revCallname", makeLiteral("Call for reviews for "+confTitle));
			substitution.put("camCallname", makeLiteral("Call for camera-ready papers for "+confTitle));
			substitution.put("notifEvtname", makeLiteral("Author notification for "+confTitle));
			substitution.put("regisEvtname", makeLiteral("Conference participant registration for "+confTitle));

			//Chairs (the pc, oc and general chairs attend the conference)
			String pcchairIRI= people.addPerson(conference.get("pc_chair").toString(),prefix,reader);
			substitution.put("pcchair", pcchairIRI);
			people.getPersonFromIRI(pcchairIRI).attendsConference();
			String occhairIRI= people.addPerson(conference.get("oc_chair").toString(),prefix,reader);
			substitution.put("occhair", occhairIRI);
			people.getPersonFromIRI(occhairIRI).attendsConference();
			String scchairIRI= people.addPerson(conference.get("sc_chair").toString(),prefix,reader);
			substitution.put("scchair", scchairIRI);
			String generalchairIRI= people.addPerson(conference.get("general_chair").toString(),prefix,reader);
			people.getPersonFromIRI(generalchairIRI).attendsConference();
			substitution.put("generalchair", generalchairIRI);

			//PC, OC, SC
			substitution.put("pc", makeIRI("pc"+confHash,prefix));
			substitution.put("pcname", makeLiteral("Program committee of "+confTitle));
			substitution.put("oc", makeIRI("oc"+confHash,prefix));
			substitution.put("ocname", makeLiteral("Organizing committee of "+confTitle));
			substitution.put("sc", makeIRI("sc"+confHash,prefix));
			substitution.put("scname", makeLiteral("Steering committee of "+confTitle));

			//PROCEEDINGS
			JSONObject proceedings = (JSONObject)conference.get("proceedings");
			substitution.put("confproceed", makeIRI("proceedings"+confHash,prefix));
			substitution.put("proceedname",  makeLiteral(proceedings.get("title").toString()));
			substitution.put("proceedvolume",  makeLiteral(proceedings.get("volume").toString()));
			substitution.put("proceedISBN",  makeLiteral(proceedings.get("ISBN").toString()));
			substitution.put("publisher", makeIRI("publisher"+proceedings.get("publisher").toString().hashCode(),prefix));
			substitution.put("publishername",  makeLiteral(proceedings.get("publisher").toString()));
			runQuery(ontologyname, "conference", queryTemplates, substitution,spEndpoint);

			//FOR EACH PC, OC, SC member
			JSONArray pcmembers= (JSONArray)conference.get("pc_members");
			for(Object pcmembername: pcmembers) {
				String pcmemberIRI= people.addPerson(pcmembername.toString(),prefix,reader);
				people.getPersonFromIRI(pcmemberIRI).forcePCMember();
				substitution.put("pcmember", pcmemberIRI);
				//this query is run later, after all reviews have been done
				//runQuery(ontologyname, "pcmember", queryTemplates, substitution,spEndpoint);
			}

			JSONArray ocmembers= (JSONArray)conference.get("oc_members");
			for(Object ocmembername: ocmembers) {
				String ocmemberIRI= people.addPerson(ocmembername.toString(),prefix,reader);
				substitution.put("ocmember", ocmemberIRI);
				runQuery(ontologyname, "ocmember", queryTemplates, substitution,spEndpoint);
			}

			JSONArray scmembers= (JSONArray)conference.get("sc_members");
			for(Object scmembername: scmembers) {
				String scmemberIRI= people.addPerson(scmembername.toString(),prefix,reader);
				substitution.put("scmember", scmemberIRI);
				runQuery(ontologyname, "scmember", queryTemplates, substitution,spEndpoint);
			}
			
			if (conference.containsKey("topics")) {
				JSONArray keywords = (JSONArray) (conference.get("topics"));
				for(Object keywordObj:keywords) {
					String keyword = keywordObj.toString().toLowerCase();
					substitution.put("keyword", makeLiteral(keyword));
					substitution.put("topic", makeIRI("topic"+keyword.hashCode(),prefix));
					runQuery(ontologyname, "conferencetopic", queryTemplates, substitution,spEndpoint);			
				}
			}


			//WORKSHOPS
			JSONArray workshops= (JSONArray)conference.get("workshops");
			for(Object workshopobj: workshops) {
				JSONObject workshop = ((JSONObject)(workshopobj));
				String workshopTitle = workshop.get("title").toString();
				int workshopHash= workshopTitle.hashCode();
				String workshopchairIRI= people.addPerson(workshop.get("chair").toString(),prefix,reader);
				String workshopIRI = makeIRI("workshop"+workshopHash,prefix);
				substitution.put("workshopchair", workshopchairIRI);
				//the workshop chair attends the conference
				people.getPersonFromIRI(workshopchairIRI).attendsConference();

				substitution.put("workshopname",makeLiteral(workshop.get("title").toString()));
				substitution.put("workshop",workshopIRI);
				if (workshop.containsKey("short_title")) {
					substitution.put("workshopshortname",makeLiteral(workshop.get("short_title").toString()));
				}
				else { 
					substitution.put("workshopshortname",makeLiteral(workshop.get("title").toString()));
				}
				substitution.put("workshoplocinst",makeIRI("location_main_venue_"+confHash,prefix));

				substitution.put("subCall", makeIRI("subCall"+workshopHash,prefix));				
				substitution.put("revCall", makeIRI("revCall"+workshopHash,prefix));
				substitution.put("camCall", makeIRI("camCall"+workshopHash,prefix));

				substitution.put("subCallname", makeLiteral("Call for papers for "+workshopTitle));
				substitution.put("revCallname", makeLiteral("Call for reviews for "+workshopTitle));
				substitution.put("camCallname", makeLiteral("Call for camera-ready papers for "+workshopTitle));
				substitution.put("notifEvtname", makeLiteral("Author nottification for "+workshopTitle));

				substitution.put("subEventInst", makeIRI("subEvt"+workshopHash,prefix));
				substitution.put("reviewEventInst", makeIRI("revEvt"+workshopHash,prefix));
				substitution.put("notifEventInst", makeIRI("notifEvt"+workshopHash,prefix));
				substitution.put("camEventInst", makeIRI("camEvt"+workshopHash,prefix));

				substitution.put("startDate", makeLiteral(workshop.get("start").toString()));
				substitution.put("endDate", makeLiteral(workshop.get("end").toString()));
				substitution.put("subDate", makeLiteral(workshop.get("submission_deadline").toString()));
				substitution.put("notifDate", makeLiteral(workshop.get("review_deadline").toString()));
				substitution.put("camDate", makeLiteral(workshop.get("camera-ready_deadline").toString()));
				
				if (workshop.containsKey("topics")) {
					JSONArray keywords = (JSONArray) (workshop.get("topics"));
					for(Object keywordObj:keywords) {
						String keyword = keywordObj.toString().toLowerCase();
						substitution.put("keyword", makeLiteral(keyword));
						substitution.put("topic", makeIRI("topic"+keyword.hashCode(),prefix));
						runQuery(ontologyname, "workshoptopic", queryTemplates, substitution,spEndpoint);			
					}
				}

				runQuery(ontologyname, "workshop", queryTemplates, substitution,spEndpoint);
			}


			//TUTORIALS
			JSONArray tutorials= (JSONArray)conference.get("tutorials");
			for(Object tutorialobj: tutorials) {
				JSONObject tutorial = ((JSONObject)(tutorialobj));
				int tutorialHash= tutorial.get("title").toString().hashCode();
				String tutorialchairIRI= people.addPerson(tutorial.get("chair").toString(),prefix,reader);
				people.getPersonFromIRI(tutorialchairIRI).attendsConference();
				String tutorialIRI = makeIRI("tutorial"+tutorialHash,prefix);
				substitution.put("tutochair", tutorialchairIRI);

				if (tutorial.containsKey("co-chair")) {
					JSONArray tutoCoChairs = (JSONArray) tutorial.get("co-chair");
					for(Object tutoCoChairObj : tutoCoChairs) {
						people.addPerson(tutoCoChairObj.toString(), prefix, reader);
					}
				}

				if (tutorial.containsKey("short_title")) {
					substitution.put("tutoshortname",makeLiteral(tutorial.get("short_title").toString()));
				}
				else { 
					substitution.put("tutoshortname",makeLiteral(tutorial.get("title").toString()));
				}

				substitution.put("tutoname",makeLiteral(tutorial.get("title").toString()));
				substitution.put("tuto",tutorialIRI);
				substitution.put("tutoabstr",makeIRI("tutoabstr"+tutorialHash,prefix));
				substitution.put("tutolocinst",makeIRI("location_main_venue",prefix));
				substitution.put("startDate", makeLiteral(tutorial.get("start").toString()));
				substitution.put("endDate", makeLiteral(tutorial.get("end").toString()));
				
				if (tutorial.containsKey("topics")) {
					JSONArray keywords = (JSONArray) (tutorial.get("topics"));
					for(Object keywordObj:keywords) {
						String keyword = keywordObj.toString().toLowerCase();
						substitution.put("keyword", makeLiteral(keyword));
						substitution.put("topic", makeIRI("topic"+keyword.hashCode(),prefix));
						runQuery(ontologyname, "tutorialtopic", queryTemplates, substitution,spEndpoint);			
					}
				}

				runQuery(ontologyname, "tutorial", queryTemplates, substitution,spEndpoint);
			}
			
			//TRACKS
			JSONArray tracks= (JSONArray)conference.get("tracks");
			for(Object trackobj: tracks) {
				JSONObject track = ((JSONObject)(trackobj));
				int trackHash= track.get("title").toString().hashCode();
				String trackchairIRI= people.addPerson(track.get("chair").toString(),prefix,reader);
				String trackIRI=makeIRI("track"+trackHash+"_"+confHash,prefix);
				substitution.put("trackchair", trackchairIRI);
				substitution.put("trackname",makeLiteral(track.get("title").toString()));
				substitution.put("track",trackIRI);
				
				if (track.containsKey("topics")) {
					JSONArray keywords = (JSONArray) (track.get("topics"));
					for(Object keywordObj:keywords) {
						String keyword = keywordObj.toString().toLowerCase();
						substitution.put("keyword", makeLiteral(keyword));
						substitution.put("topic", makeIRI("topic"+keyword.hashCode(),prefix));
						runQuery(ontologyname, "tracktopic", queryTemplates, substitution,spEndpoint);			
					}
				}

				runQuery(ontologyname, "track", queryTemplates, substitution,spEndpoint);
			}

			//Social events
			JSONArray events= (JSONArray)conference.get("social_events");
			for(Object eventobj: events) {
				JSONObject event = ((JSONObject)(eventobj));
				int eventHash= event.get("name").toString().hashCode();
				substitution.put("eventname",makeLiteral(event.get("name").toString()));
				substitution.put("eventloc",makeLiteral(event.get("location").toString()));
				substitution.put("event",makeIRI("event"+eventHash+"_"+confHash,prefix));
				substitution.put("eventlocinst",makeIRI("location"+event.get("location").toString().hashCode(),prefix));
				substitution.put("startDate", makeLiteral(event.get("start").toString()));
				substitution.put("endDate", makeLiteral(event.get("end").toString()));

				String eventType = event.get("type").toString();
				if (eventType.equals("Reception")) {
					runQuery(ontologyname, "reception", queryTemplates, substitution,spEndpoint);
				}
				else if (eventType.equals("Banquet")) {
					runQuery(ontologyname, "galadinner", queryTemplates, substitution,spEndpoint);
				}
				else if (eventType.equals("Excursion")) {
					runQuery(ontologyname, "trip", queryTemplates, substitution,spEndpoint);
				}
			}

			//INVITED TALKS
			JSONArray invitedTalks = (JSONArray)conference.get("invited_talks");
			for(Object invTalkObj: invitedTalks) {
				JSONObject invitedTalk = ((JSONObject)(invTalkObj));
				int invTalkHash= invitedTalk.get("title").toString().hashCode();				
				String invitedSpeakerIRI= people.addPerson(invitedTalk.get("speaker").toString(),prefix,reader);
				people.getPersonFromIRI(invitedSpeakerIRI).attendsConference();
				substitution.put("inv",invitedSpeakerIRI);
				substitution.put("invtalkname",makeLiteral(invitedTalk.get("title").toString()));
				substitution.put("invtalk",makeIRI("invTalk"+invTalkHash,prefix));
				substitution.put("invabstr",makeIRI("invAbstr"+invTalkHash,prefix));
				//PAPER TOPICS
				if (invitedTalk.containsKey("topics")) {
					JSONArray keywords = (JSONArray) (invitedTalk.get("topics"));
					for(Object keywordObj:keywords) {
						String keyword = keywordObj.toString().toLowerCase();
						substitution.put("keyword", makeLiteral(keyword));
						substitution.put("topic", makeIRI("topic"+keyword.hashCode(),prefix));
						runQuery(ontologyname, "invitedtalktopic", queryTemplates, substitution,spEndpoint);			
					}
				}

				runQuery(ontologyname, "invitedtalk", queryTemplates, substitution,spEndpoint);
			}

			//Administrator
			String adminIRI = people.addPerson(conference.get("administrator").toString(), prefix,reader);
			substitution.put("admin", adminIRI);
			runQuery(ontologyname, "administrator", queryTemplates, substitution,spEndpoint);


			//PAPER FILE
			System.out.println("PAPER TIME");
			JSONArray papers = (JSONArray) (parser.parse(new FileReader(conference.get("paper_file").toString())));
			for(Object paperObj: papers) {
				JSONObject paper = ((JSONObject)(paperObj));
				String paperType = paper.get("type").toString().toLowerCase();
				String paperTitle = paper.get("title").toString();
				int paperHashn = paperTitle.hashCode();
				String paperId= paper.get("id").toString();
				String paperHash = paperHashn+paperId;
				String campapIRI= makeIRI("paper"+paperHash+"_v2",prefix);
				String papV0IRI= makeIRI("paper"+paperHash+"_v0",prefix);
				String paperIRI= makeIRI("paper"+paperHash,prefix);
				String paperDecision = paper.get("decision").toString();

				substitution.put("pap", paperIRI);
				substitution.put("paptitle", makeLiteral(paperTitle));
				substitution.put("campap", campapIRI);
				substitution.put("papv0", papV0IRI);
				substitution.put("dec", makeIRI("dec"+paperHash,prefix));
				substitution.put("papabstr", makeIRI("abstr"+paperHash,prefix));
				substitution.put("papId", makeLiteral(paperId));

				// get authors IRIs
				ArrayList<String> paperAuthorsIRIs = new ArrayList<String>();
				JSONArray authorsJson = (JSONArray)paper.get("authors");
				for (Object authObj : authorsJson) {
					String authorIRI = people.addPerson(authObj.toString(),prefix,reader);
					paperAuthorsIRIs.add(authorIRI);
				}

				//Co authors
				for(int i = 1; i <paperAuthorsIRIs.size();i++) {
					substitution.put("auth", paperAuthorsIRIs.get(i));
					runQuery(ontologyname, "coauthor", queryTemplates, substitution,spEndpoint);
				}

				// First author
				substitution.put("auth", paperAuthorsIRIs.get(0));

				//paper type
				if (paperType.equals("demo")) {
					runQuery(ontologyname, "demopaper", queryTemplates, substitution,spEndpoint);
				}
				else if (paperType.equals("poster")) {
					runQuery(ontologyname, "posterpaper", queryTemplates, substitution,spEndpoint);
				}
				else {
					runQuery(ontologyname, "regularpaper", queryTemplates, substitution,spEndpoint);
				}

				//paper decision
				//ACCEPTED PAPER
				if(paperDecision.equals("accept")) {
					runQuery(ontologyname, "acceptedpaper", queryTemplates, substitution,spEndpoint);
					
					//If not a demo or poster paper, then presentation
					if (paper.containsKey("presenter")) {		
						String presenterIRI = people.addPerson(paper.get("presenter").toString(),prefix,reader);
						people.getPersonFromIRI(presenterIRI).attendsConference();
						substitution.put("pres", presenterIRI);
						substitution.put("presentation", makeIRI("presentation"+paperHash,prefix));
						substitution.put("presentationname", makeLiteral("Presentation of "+paperTitle));
						substitution.put("diapo", makeIRI("diapo"+paperHash,prefix));
						runQuery(ontologyname, "presentation", queryTemplates, substitution,spEndpoint);
					}
					// pour chaque auteur: run cocameraauthor
					for(int i = 1; i <paperAuthorsIRIs.size();i++) {
						substitution.put("auth", paperAuthorsIRIs.get(i));
						runQuery(ontologyname, "cocameraauthor", queryTemplates, substitution,spEndpoint);
					}
					
					// if an accepted poster
					if(paperType.equals("poster")) {
						substitution.put("poster", makeIRI("poster"+paperHash,prefix));
						runQuery(ontologyname,"acceptedposter",queryTemplates,substitution,spEndpoint);
					}
				}
				// REJECTED PAPER
				else {
					//System.out.println(substitution);
					runQuery(ontologyname, "rejectedpaper", queryTemplates, substitution,spEndpoint);
				}
				
				//PAPER TOPICS
				if (paper.containsKey("keywords")) {
					JSONArray keywords = (JSONArray) (paper.get("keywords"));
					for(Object keywordObj:keywords) {
						String keyword = keywordObj.toString().toLowerCase();
						substitution.put("keyword", makeLiteral(keyword));
						substitution.put("topic", makeIRI("topic"+keyword.hashCode(),prefix));
						if(paperDecision.equals("accept")) {
							runQuery(ontologyname, "acceptedpapertopic", queryTemplates, substitution,spEndpoint);
							if (paperType.equals("regular")) {
								runQuery(ontologyname, "presentationtopic", queryTemplates, substitution,spEndpoint);	
							}							
						}
						
						runQuery(ontologyname, "papertopic", queryTemplates, substitution,spEndpoint);			
					}
				}

				//REVIEWS
				if (paper.containsKey("reviews")) {
					JSONArray reviews = (JSONArray) (paper.get("reviews"));
					//System.out.println("reviews "+reviews.size());
					double reviewScore=0;
					for(Object reviewObj: reviews) {
						JSONObject review = ((JSONObject)(reviewObj));
						String reviewId = review.get("id").toString();
						String reviewRate = review.get("rate").toString();
						int reviewHash= reviewId.hashCode();
						JSONArray reviewers = (JSONArray) review.get("author");

						//if only one reviewer, he was assigned to the paper

						String reviewerIRI = people.addPerson(reviewers.get(0).toString(), prefix,reader);
						people.getPersonFromIRI(reviewerIRI).forcePCMember();
						substitution.put("rev", reviewerIRI);
						substitution.put("review", makeIRI("review"+paperHash+"_"+reviewHash,prefix));
						substitution.put("reviewname", makeLiteral(reviewId + " of "+paperTitle));

						runQuery(ontologyname, "assignedreviewer",queryTemplates, substitution, spEndpoint);

						//is a metareview also a review ? (for all ontologies) Let's say yes

						if (reviewId.toLowerCase().equals("metareview")) {
							runQuery(ontologyname, "metareview", queryTemplates, substitution,spEndpoint);
						}
						else {
							runQuery(ontologyname, "review", queryTemplates, substitution,spEndpoint);						
						}
						
						// review rating
						if(reviewRate.equals("accept")) {
							runQuery(ontologyname, "acceptreview", queryTemplates, substitution,spEndpoint);	
							reviewScore +=2;
						}
						else if(reviewRate.equals("weak accept")) {
							runQuery(ontologyname, "weakacceptreview", queryTemplates, substitution,spEndpoint);
							reviewScore +=1;
						}
						else if(reviewRate.equals("weak reject")) {
							runQuery(ontologyname, "weakrejectreview", queryTemplates, substitution,spEndpoint);	
							reviewScore +=-1;
						}
						else if(reviewRate.equals("reject")) {
							runQuery(ontologyname, "rejectreview", queryTemplates, substitution,spEndpoint);	
							reviewScore +=-2;
						}

						// if 2 reviewers, the 2nd one was invited by the first one.
						if(reviewers.size()==2) {
							String extRevIRI = people.addPerson(reviewers.get(1).toString(), prefix,reader);
							substitution.put("extrev", extRevIRI);
							runQuery(ontologyname, "invitereviewer",queryTemplates, substitution, spEndpoint);
						}
					}
					
					substitution.put("revRating", makeIRI("revRate"+paperHash,prefix));
					reviewScore = reviewScore/reviews.size();
					if (reviewScore > 1) {
						runQuery(ontologyname, "acceptrating",queryTemplates, substitution, spEndpoint);
					}
					else if (reviewScore > 0) {
						runQuery(ontologyname, "weakacceptrating",queryTemplates, substitution, spEndpoint);
					}
					else if (reviewScore > -1) {
						runQuery(ontologyname, "weakrejectrating",queryTemplates, substitution, spEndpoint);
					}
					else {
						runQuery(ontologyname, "rejectrating",queryTemplates, substitution, spEndpoint);
					}
					
					
				}


				

			}

			System.out.println("PEOPLE TIME");
			
			
			for (Person person: people.getPeople()) {
			
//				if (person.hasNotSureField()) {
//					for(String s : person.getNotSureStrings()) {
//						System.out.println(person.getFirstName()+ " - "+ person.getLastName()+ "  - not sure: "+s + "\n"
//								+ "Press 1 to attach notsure to firstName\n"
//								+ "Press 2 to attact not sure to lastName\n"
//								+ "Another key to not attach it");
//						String n = reader.next();
//						if (n.equals("1")) {
//							person.attachNotSureToFirstName(s);
//						}
//						else if (n.equals("2")) {
//							person.attachNotSureToLasName(s);
//						}
//					}
//				}
				substitution.put("pers", person.IRI);
				substitution.put("member", person.IRI);
				substitution.put("persname", makeLiteral(person.getName()));
				substitution.put("persfname", makeLiteral(person.getFirstName()));
				substitution.put("perslname", makeLiteral(person.getLastName()));
				runQuery(ontologyname, "person", queryTemplates, substitution,spEndpoint);

				if(person.isAttendee() && person.isEarlyRegistration()) {
					runQuery(ontologyname, "earlyparticipant", queryTemplates, substitution,spEndpoint);
				}
				else if(person.isAttendee() &&  !person.isEarlyRegistration()) {
					runQuery(ontologyname, "lateparticipant", queryTemplates, substitution,spEndpoint);
				}
				
				if(person.isPCMember()) {
					substitution.put("pcmember", person.IRI);
					runQuery(ontologyname, "pcmember", queryTemplates, substitution,spEndpoint);
				}
				substitution.put("orga",person.getOrganisation().getIRI());
				substitution.put("organame",makeLiteral(person.getOrganisation().getName()));
				if(person.isStudent()) {
					runQuery(ontologyname, "student", queryTemplates, substitution,spEndpoint);
				}
				else {
					runQuery(ontologyname, "employee", queryTemplates, substitution,spEndpoint);
				}
			}
		//	people.printContent();
			for (Organisation orga: organisations.values()) {
				substitution.put("orga",orga.getIRI());
				substitution.put("organame",makeLiteral(orga.getName()));
				if(orga.isCompany()) {
					runQuery(ontologyname, "company", queryTemplates, substitution,spEndpoint);	
				}
				else if(orga.isUniversity()) {
					//System.out.println(orga.getIRI());
					//System.out.println(makeLiteral(orga.getName()));
					runQuery(ontologyname, "university", queryTemplates, substitution,spEndpoint);	
				}
			}
		

		} catch (IOException | ParseException | IncompleteSubstitutionException | SparqlQueryMalFormedException | SparqlEndpointUnreachableException e) {
			e.printStackTrace();
		}		
	}

	public static String makeIRI(String suffix, String prefix) {
		return "<"+prefix+suffix+">";
	}
	public static String makeLiteral(String content) {
		return "\""+content+"\"";
	}

	public static void runQuery(String ontologyName, String queryName, 
			Map<String,FolderManager> queryTemplates, Map<String, String> substitution,
			SparqlProxy spEndpoint) throws IncompleteSubstitutionException, SparqlQueryMalFormedException, SparqlEndpointUnreachableException {
		if(queryTemplates.get(ontologyName).getTemplateQueries().containsKey(queryName)) {
			String query= queryTemplates.get(ontologyName).getTemplateQueries().get(queryName).substitute(substitution);
			
			//System.out.println(ontologyName+" - "+ queryName);
			//System.out.println(query);
			spEndpoint.postSparqlUpdateQuery(query);
		}	
	}

	public static void basicInference(SparqlProxy sp,Map<String,FolderManager> queryTemplates) throws NotAFolderException, SparqlQueryMalFormedException, SparqlEndpointUnreachableException {
		queryTemplates.put("inference", new FolderManager("query_templates/inference/"));
		queryTemplates.get("inference").loadQueries();
		String query="";
		query = queryTemplates.get("inference").getQueries().get("inverse");
		sp.postSparqlUpdateQuery(query);
		query = queryTemplates.get("inference").getQueries().get("subproperty");
		sp.postSparqlUpdateQuery(query);
		query = queryTemplates.get("inference").getQueries().get("domain");
		sp.postSparqlUpdateQuery(query);
		query = queryTemplates.get("inference").getQueries().get("range");
		sp.postSparqlUpdateQuery(query);
		query = queryTemplates.get("inference").getQueries().get("subclass");
		sp.postSparqlUpdateQuery(query);
		for(String infQuery : queryTemplates.get("inference").getQueries().values()) {
			sp.postSparqlUpdateQuery(infQuery);
		}
	}
}
