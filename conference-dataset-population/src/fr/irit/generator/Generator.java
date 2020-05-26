package fr.irit.generator;


import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import com.fasterxml.jackson.databind.JsonNode;

import fr.irit.melodi.sparql.exceptions.IncompleteSubstitutionException;
import fr.irit.melodi.sparql.exceptions.NotAFolderException;
import fr.irit.melodi.sparql.files.FolderManager;
import fr.irit.melodi.sparql.proxy.SparqlProxy;
import fr.irit.melodi.sparql.query.exceptions.SparqlEndpointUnreachableException;
import fr.irit.melodi.sparql.query.exceptions.SparqlQueryMalFormedException;
import fr.irit.population.Organisation;
import fr.irit.population.Person;

public class Generator {

	public static void main(String[] args) {
		SparqlProxy spDBpedia = SparqlProxy.getSparqlProxy("http://dbpedia.org/sparql/");
		SparqlProxy spScholar = SparqlProxy.getSparqlProxy("http://www.scholarlydata.org/sparql/");
		SparqlProxy spHAL = SparqlProxy.getSparqlProxy("http://sparql.archives-ouvertes.fr/sparql/");
		JSONParser parser = new JSONParser();
		int offsetPerson = 0;
		int offsetOrga= 0;
		int offsetConf=0;
		int offsetPlace=0;
		int offsetPaper = 0;

		Map<String,FolderManager> queryTemplates = new HashMap<>();
		Map<String, String> substitution = new HashMap<>();
		Map<String,JSONObject> conferencesMap = new HashMap<>();
		ArrayList<JSONObject> conferences = new ArrayList<JSONObject>();
		ArrayList<String> publishers = new ArrayList <String>();
		ArrayList<String> topics = new ArrayList <String>();
		String [] trackNames = {"Research track", "Posters and Demos track", "Resource track", "In-use track", "Industry track", "Journal track"};
		ArrayList<String> tracks = new ArrayList(Arrays.asList(trackNames));
		try {
			queryTemplates.put("dbpedia", new FolderManager("scholarly_data_queries/dbpedia/"));
			queryTemplates.get("dbpedia").loadQueries();
			queryTemplates.put("hal", new FolderManager("scholarly_data_queries/hal/"));
			queryTemplates.get("hal").loadQueries();
			queryTemplates.put("scholarlydata", new FolderManager("scholarly_data_queries/scholarlydata/"));
			queryTemplates.get("scholarlydata").loadQueries();

			JSONArray conferencesArray = (JSONArray) (parser.parse(new FileReader("scholarly_data_queries/conference_and_workshop.json")));
			for(Object confObj: conferencesArray) {
				JSONObject confjson = (JSONObject)confObj;
				JSONObject confInfoJson = (JSONObject)confjson.get("info");
				conferencesMap.put(confInfoJson.get("venue").toString(), confInfoJson);
			}
			for(JSONObject c: conferencesMap.values()) {
				conferences.add(c);
			}

			// get 25 logo/confURLs
			substitution.put("numURL", "25");
			ArrayList<JsonNode> retURL = spDBpedia.getResponse(queryTemplates.get("dbpedia").getTemplateQueries().get("logo").substitute(substitution));
			ArrayList<String> confURLs = new ArrayList<>();
			ArrayList<String> logoURLs = new ArrayList<>();
			Iterator<JsonNode> retURLit = retURL.iterator();
			while (retURLit.hasNext()) {
				JsonNode nexturl = retURLit.next();
				confURLs.add(nexturl.get("confurl").get("value").toString().replaceAll("\"", ""));
				logoURLs.add(nexturl.get("logourl").get("value").toString().replaceAll("\"", ""));
			}

			// get 25 publishers
			substitution.put("numPublisher", "25");
			ArrayList<JsonNode> retpub = spHAL.getResponse(queryTemplates.get("hal").getTemplateQueries().get("publisher").substitute(substitution));
			Iterator<JsonNode> retpubit = retpub.iterator();
			while (retpubit.hasNext()) {
				JsonNode nextpub = retpubit.next();
				String pub = nextpub.get("publisher").get("value").toString().replaceAll("\"", "");
				if(!pub.equals("")) {
					publishers.add(pub);
				}				
			}

			ArrayList<JsonNode> retTopic = spScholar.getResponse(queryTemplates.get("scholarlydata").getQueries().get("topic"));
			Iterator<JsonNode> retTopicit = retTopic.iterator();
			while (retTopicit.hasNext()) {
				JsonNode nextTopic = retTopicit.next();
				String top = nextTopic.get("y").get("value").toString().replaceAll("\"", "");
				if(!top.equals("")) {
					topics.add(top);
				}
				
			}


			//*****************FOR EACH CONFERENCE************************************
			for (int nConf=0; nConf<25; nConf++) {
				//HashMap<String,Person> confpeople = new HashMap<>();
				HashMap<String,ArrayList<Person>> peoplePerOrga = new HashMap<>();
				Map<String,Person> peopleMap = new HashMap<>();
				ArrayList<Person> persons = new ArrayList<Person>();
				ArrayList<Person> pcMembers = new ArrayList<Person>();
				ArrayList<Person> externalReviewers = new ArrayList<Person>();
				ArrayList<String> locations = new ArrayList<String>();
				ArrayList <String> titles = new ArrayList<String>();
				Random rand = new Random();
				ArrayList<Organisation> organisations = new ArrayList<Organisation>();

				// acceptance rate entre 10 er 70%					
				double acceptRate= rand.nextDouble()*0.6+0.1;
				double posterAcceptRate= rand.nextDouble()*0.6+0.4;

				//petite ou grosse conference
				int confCoeff = rand.nextInt(10)+1; //if the conference is small on a scale from 1 (very small) to 10 (very big)
				System.out.println("Conference n"+nConf+" of size "+confCoeff+"/10");

				//set Of titles
				int nbPapers= 40+100*(confCoeff-1)+rand.nextInt(5*confCoeff);
				int nbTuto = 1+1*(confCoeff-1)+rand.nextInt(2);
				int nbInvitedTalks = 3+ rand.nextInt(3);
				int nbTitles = nbPapers+nbTuto+nbInvitedTalks;
				substitution.put("numPaper", ""+nbTitles);
				substitution.put("offsetPaper", ""+offsetPaper);
				offsetPaper +=nbTitles;
				int localOffsetTitle = 0;
				ArrayList<JsonNode> retTitle = spHAL.getResponse(queryTemplates.get("hal").getTemplateQueries().get("paper").substitute(substitution));
				Iterator<JsonNode> retTitleIt = retTitle.iterator();
				while (retTitleIt.hasNext()) {
					JsonNode nextTitle = retTitleIt.next();
					String title= (nextTitle.get("paper").get("value").toString().replaceAll("\"", "").replaceAll("\\$", ""));
					if(!title.equals("")) {
						titles.add(title);
					}					
				}



				//set Of 10 places
				offsetPlace += 10;
				substitution.put("numPlace", "1");
				substitution.put("offsetPlace", ""+offsetPlace);
				ArrayList<JsonNode> retLoc = spDBpedia.getResponse(queryTemplates.get("dbpedia").getTemplateQueries().get("place").substitute(substitution));
				Iterator<JsonNode> retLocIt = retLoc.iterator();
				while (retLocIt.hasNext()) {
					JsonNode nextLoc = retLocIt.next();
					String locName= (nextLoc.get("place").get("value").toString().replaceAll("\"", ""));
					if (!locName.equals("")) {
						locations.add(locName);
					}					
				}


				//Set d'organisations entre 30 et 230
				int nbOrga = 30+20*(confCoeff-1)+rand.nextInt(3*confCoeff);
				substitution.put("numOrganisation", ""+nbOrga);
				substitution.put("offsetOrganisation", ""+offsetOrga);
				offsetOrga += nbOrga;
				ArrayList<JsonNode> retOrga = spScholar.getResponse(queryTemplates.get("scholarlydata").getTemplateQueries().get("organisation").substitute(substitution));
				Iterator<JsonNode> retOrgait = retOrga.iterator();
				while (retOrgait.hasNext()) {
					JsonNode nextOrga = retOrgait.next();
					String orgaName= (nextOrga.get("orgname").get("value").toString().replaceAll("\"", ""));
					if(!orgaName.equals("")) {
						organisations.add(new Organisation(orgaName));
						peoplePerOrga.put(orgaName,new ArrayList<Person>());
					}					
				}

				//set de personnes entre 300 et 2000
				int nbPers = 300+170*(confCoeff-1)+rand.nextInt(30*confCoeff);
				substitution.put("numPerson", ""+nbPers);
				substitution.put("offsetPerson", ""+offsetPerson);
				offsetPerson+=nbPers;
				ArrayList<JsonNode> retPers = spDBpedia.getResponse(queryTemplates.get("dbpedia").getTemplateQueries().get("person").substitute(substitution));
				Iterator<JsonNode> retPersit = retPers.iterator();
				while (retPersit.hasNext()) {
					JsonNode nextPers = retPersit.next();
					String dbpediaIRI= nextPers.get("x").get("value").toString().replaceAll("\"", "");
					String name = nextPers.get("name").get("value").toString().replaceAll("\"", "");
					if (!peopleMap.containsKey(dbpediaIRI)) {
						Organisation orga = organisations.get(rand.nextInt(nbOrga));
						Person p = new Person(dbpediaIRI,name,orga);
						peopleMap.put(dbpediaIRI, p);
						peoplePerOrga.get(orga.getName()).add(p);
						persons.add(p);
					}					
				}


				JSONObject conference = new JSONObject();
				String confTitle = conferences.get(offsetConf).get("venue").toString();
				conference.put("title", confTitle);
				if (conferences.get(offsetConf).containsKey("acronym")) {
					conference.put("short_title", conferences.get(offsetConf).get("acronym").toString());
				}
				else {
					conference.put("short_title", conferences.get(offsetConf).get("venue").toString());
				}
				offsetConf ++;

				//urls site et logo
				conference.put("url", confURLs.get(nConf));
				conference.put("logo_url", logoURLs.get(nConf));

				//topics
				JSONArray confTopics = new JSONArray();
				int nbConfTopics= rand.nextInt(20)+20;
				for(int i = 0; i <nbConfTopics; i++) {
					confTopics.add(topics.get(rand.nextInt(topics.size())));
				}
				conference.put("topics", confTopics);

				//Dates
				Date endDate = new Date();
				Date beginDate = endDate.takeOffDays(3);
				Date registrationDate= endDate.takeOffDays(10);
				Date cameraDate = endDate.takeOffDays(100);
				Date reviewDate = endDate.takeOffDays(125);
				Date submissionDate = endDate.takeOffDays(155);
				Date absDate = endDate.takeOffDays(172);

				beginDate.setTime(9);
				endDate.setTime(18);

				conference.put("abstract_deadline", absDate.toString());
				conference.put("submission_deadline", submissionDate.toString());
				conference.put("review_deadline", reviewDate.toString());
				conference.put("camera-ready_deadline", cameraDate.toString());
				conference.put("registration_deadline", registrationDate.toString());

				conference.put("start", beginDate.toString());
				conference.put("end", endDate.toString());

				//conf location
				conference.put("location", locations.get(rand.nextInt(locations.size())));

				int minReview= (int)(Math.random()*2)+2;
				conference.put("num_reviews", minReview);				

				//social events: welcome, gala, sometimes excursion
				JSONArray socialEvents = new JSONArray();
				JSONObject welcomeReception =new JSONObject();
				welcomeReception.put("name", "Welcome Reception");
				welcomeReception.put("type", "Reception");
				beginDate.setTime(19);
				welcomeReception.put("start", beginDate.toString());
				beginDate.setTime(21);
				welcomeReception.put("end", beginDate.toString());
				welcomeReception.put("location", locations.get(rand.nextInt(locations.size())));
				socialEvents.add(welcomeReception);

				JSONObject gala =new JSONObject();
				gala.put("name", "Gala Dinner");
				gala.put("type", "Banquet");
				Date galaDate = endDate.takeOffDays(1);
				galaDate.setTime(19);
				gala.put("start", galaDate.toString());
				galaDate.setTime(23);
				gala.put("end", galaDate.toString());
				gala.put("location", locations.get(rand.nextInt(locations.size())));
				socialEvents.add(gala);

				JSONObject trip =new JSONObject();
				trip.put("name", "Excursion");
				trip.put("type", "Excursion");
				Date tripDate = endDate.takeOffDays(2);
				tripDate.setTime(18);
				trip.put("start", tripDate.toString());
				tripDate.setTime(20);
				trip.put("end", tripDate.toString());
				trip.put("location", locations.get(rand.nextInt(locations.size())));
				socialEvents.add(trip);
				conference.put("social_events", socialEvents);



				//workshops (dates) entre 1 et 18 workshops par conf
				int nbWorkshops = 1+2*(confCoeff-1)+rand.nextInt(2);
				JSONArray workshops = new JSONArray();
				for (int i = offsetConf; i < offsetConf + nbWorkshops; i++) {
					JSONObject workshop = new JSONObject();
					workshop.put("title", conferences.get(i).get("venue"));
					if (conferences.get(i).containsKey("acronym")) {
						workshop.put("short_title", conferences.get(i).get("acronym"));
					}
					else {
						workshop.put("short_title", conferences.get(i).get("venue"));
					}
					Date workshopDate = beginDate.takeOffDays(rand.nextInt(2)+1);
					Date workshopSubDate = workshopDate.takeOffDays(rand.nextInt(30)+80);
					workshop.put("abstract_deadline", workshopSubDate.takeOffDays(7).toString());
					workshop.put("submission_deadline", workshopSubDate.toString());
					workshop.put("review_deadline", workshopDate.takeOffDays(30).toString());
					workshop.put("camera-ready_deadline", workshopDate.takeOffDays(10+rand.nextInt(10)).toString());
					workshopDate.setTime(9);
					workshop.put("start", workshopDate.toString());
					workshopDate.setTime(17);
					workshop.put("end", workshopDate.toString());
					//topics
					JSONArray wSTopics = new JSONArray();
					int nbWSTopics= rand.nextInt(10)+3;
					for(int j = 0; j <nbWSTopics; j++) {
						wSTopics.add(topics.get(rand.nextInt(topics.size())));
					}
					workshop.put("topics", wSTopics);
					//chair
					workshop.put("chair", persons.get(rand.nextInt(persons.size())).getName());
					workshops.add(workshop);
				}
				conference.put("workshops", workshops);
				offsetConf+=nbWorkshops;

				
				//proceedings
				JSONObject proceedings = new JSONObject();
				proceedings.put("publisher", publishers.get(nConf));
				proceedings.put("title", "Proceedings of "+confTitle);
				proceedings.put("ISBN", rand.nextInt(1000)+"-"+rand.nextInt(1000)
				+"-"+rand.nextInt(1000)+"-"+rand.nextInt(1000)+"-"+rand.nextInt(1000));
				proceedings.put("volume", rand.nextInt(20));
				conference.put("proceedings", proceedings);

				//invitedtalks

				JSONArray invitedTalks = new JSONArray();
				for (int i = 0; i < nbInvitedTalks; i++) {
					JSONObject invTalk = new JSONObject();
					Person p = persons.get(rand.nextInt(persons.size()));
					p.forceNotStudent();
					p.attendsConference();
					invTalk.put("speaker", p.getName());
					invTalk.put("title", titles.get(localOffsetTitle));
					localOffsetTitle++;
					// topics
					JSONArray invTopics = new JSONArray();
					int nbInvTopics= rand.nextInt(3)+2;
					for(int j = 0; j <nbInvTopics; j++) {
						invTopics.add(topics.get(rand.nextInt(topics.size())));
					}
					invTalk.put("topics", invTopics);

					invitedTalks.add(invTalk);
				}
				conference.put("invited_talks", invitedTalks);

				//tutorials (dates) entre 1 et 11 tutos par conf
				JSONArray tutorials = new JSONArray();
				for (int i = 0; i < nbTuto; i++) {
					JSONObject tutorial = new JSONObject();
					Date tutoDate = beginDate.takeOffDays(rand.nextInt(2)+1);
					tutoDate.setTime(rand.nextInt(8)+8);
					tutorial.put("start", tutoDate.toString());
					tutoDate.setTime(17);
					tutorial.put("end", tutoDate.toString());
					//title
					tutorial.put("title", titles.get(localOffsetTitle));
					localOffsetTitle++;

					//topics
					JSONArray tutoTopics = new JSONArray();
					int nbTutoTopics= rand.nextInt(10)+3;
					for(int j = 0; j <nbTutoTopics; j++) {
						tutoTopics.add(topics.get(rand.nextInt(topics.size())));
					}
					tutorial.put("topics", tutoTopics);
					// tutochairs
					tutorial.put("chair", persons.get(rand.nextInt(persons.size())).getName());
					JSONArray tutocochairs = new JSONArray();
					int nbTutoCochairs= rand.nextInt(5)+1;
					for(int j = 0; j <nbTutoCochairs; j++) {
						tutocochairs.add(persons.get(rand.nextInt(persons.size())).getName());
					}
					tutorial.put("co-chair",tutocochairs);
					tutorials.add(tutorial);
				}
				conference.put("tutorials", tutorials);


				//general chair
				Person gchair =  persons.get(rand.nextInt(persons.size()));
				gchair.forceNotStudent();
				conference.put("general_chair", gchair.getName());

				//admin
				Person admin =  persons.get(rand.nextInt(persons.size()));
				admin.forceNotStudent();
				conference.put("administrator", admin.getName());

				// Choisir les 50 à 500 pc members (non student)
				int nbPCMembers = 50+50*(confCoeff-1)+rand.nextInt(3*confCoeff);
				JSONArray pcMembersJson = new JSONArray();
				for (int i = 0; i < nbPCMembers; i ++) {
					Person p= persons.get(rand.nextInt(persons.size()));
					p.forcePCMember();
					pcMembers.add(p);
					pcMembersJson.add(p.getName());
				}
				conference.put("pc_members", pcMembersJson);
				conference.put("pc_chair", pcMembersJson.get(rand.nextInt(nbPCMembers)));

				// choisir 20 à 120 personnes oc member
				int nbOCMembers = 20+10*(confCoeff-1)+rand.nextInt(3*confCoeff);
				JSONArray ocMembersJson = new JSONArray();
				for (int i = 0; i < nbOCMembers; i ++) {
					Person p= persons.get(rand.nextInt(persons.size()));
					p.forceOCMember();
					ocMembersJson.add(p.getName());
				}
				conference.put("oc_members", ocMembersJson);
				conference.put("oc_chair", ocMembersJson.get(rand.nextInt(nbOCMembers)));

				// chosir 15 à 55 personnes sc member
				int nbSCMembers = 15+5*(confCoeff-1)+rand.nextInt(3*confCoeff);
				JSONArray scMembersJson = new JSONArray();
				for (int i = 0; i < nbSCMembers; i ++) {
					Person p= persons.get(rand.nextInt(persons.size()));
					p.forceSCMember();
					scMembersJson.add(p.getName());
				}
				conference.put("sc_members", scMembersJson);
				conference.put("sc_chair", scMembersJson.get(rand.nextInt(nbSCMembers)));

				//tracks
				JSONArray tracksArray = new JSONArray();
				int nbTracks = (int)(confCoeff *(2.0-(20.0-tracks.size())/9.0) + (20.0-tracks.size())/9.0);
				System.out.println(nbTracks + " tracks");
				for (int i = 0; i < nbTracks && i < tracks.size(); i ++) {
					JSONObject track= new JSONObject();
					Person trackchair = pcMembers.get(rand.nextInt(pcMembers.size()));
					Person trackcochair = pcMembers.get(rand.nextInt(pcMembers.size()));
					track.put("title", tracks.get(i));
					track.put("chair", trackchair.getName());
					track.put("co-chair", trackcochair.getName());
					tracksArray.add(track);
				}
				conference.put("tracks", tracksArray);


				//set de papiers soumis (entre 40 et 1100)

				JSONArray paperArray = new JSONArray();
				for(int i = 0; i< nbPapers ; i ++) {
					JSONObject paper = new JSONObject();
					paper.put("title", titles.get(localOffsetTitle));
					localOffsetTitle++;

					//authors
					ArrayList<Person> authList = new ArrayList<Person>();
					JSONArray authors = new JSONArray();
					ArrayList<Organisation> orgas= new ArrayList<Organisation>();
					// institutions: entre 1 (40%), 2(30%), 3 (17%), 4 (7%), 5 (5%) 6(2%)
					int nbInst;
					double instRand = rand.nextDouble();
					if (instRand <=0.4) {
						nbInst = 1;
					}
					else if(instRand <= 0.7) {
						nbInst = 2;
					}
					else if (instRand <=0.87) {
						nbInst = 3;
					}
					else if (instRand <=0.94) {
						nbInst = 4;
					}
					else if (instRand <0.98) {
						nbInst = 5;
					}
					else {
						nbInst = 6;
					}

					// auteurs: entre 1 (6%), 2 (17%), 3 (29%), 4 (26%), 5 (9%), 6 (8%) ou 7-10 (2%)
					int nbAuth ;
					double instAuth = rand.nextDouble();
					if (instAuth <=0.06) {
						nbAuth = 1;
					}
					else if(instAuth <= 0.23) {
						nbAuth = 2;
					}
					else if (instAuth <=0.52) {
						nbAuth = 3;
					}
					else if (instAuth <=0.78) {
						nbAuth = 4;
					}
					else if (instAuth <=0.87) {
						nbAuth = 5;
					}
					else if (instAuth <0.95) {
						nbAuth = 6;
					}
					else {
						nbAuth = rand.nextInt(4)+7;
					}

					// 50% de papiers avec un pc member
					if(rand.nextBoolean()) {
						Person pcmember = pcMembers.get(rand.nextInt(pcMembers.size()));
						while(pcmember.getName().equals(gchair.getName())) {
							pcmember = pcMembers.get(rand.nextInt(pcMembers.size()));
						}
						authList.add(pcmember);
						orgas.add(pcmember.getOrganisation());
					}

					int peoplePerOrgaSize = 0;
					while(orgas.size()<nbInst) {
						Organisation org = organisations.get(rand.nextInt(organisations.size()));
						if(!orgas.contains(org) && !peoplePerOrga.get(org.getName()).isEmpty()) {
							orgas.add(org);
							peoplePerOrgaSize+=peoplePerOrga.get(org.getName()).size(); 
						}						
					}
					if(peoplePerOrgaSize<nbAuth) {//if there are less people possibilities than desired authors
						nbAuth=peoplePerOrgaSize;
					}

					if(nbInst>=nbAuth) {
						for (int j = 0; j<nbAuth; j++) {
							String orgaName = orgas.get(j).getName();
							Person p = peoplePerOrga.get(orgaName).get(rand.nextInt(peoplePerOrga.get(orgaName).size()));
							while(p.getName().equals(gchair.getName())) {
								p = peoplePerOrga.get(orgaName).get(rand.nextInt(peoplePerOrga.get(orgaName).size()));
							}
							authList.add(p);
						}						
					}
					else {
						while(authList.size()<nbAuth) {
							int orgaIndex= rand.nextInt(orgas.size());
							String orgaName = orgas.get(orgaIndex).getName();
							Person p = peoplePerOrga.get(orgaName).get(rand.nextInt(peoplePerOrga.get(orgaName).size()));
							if (p.getName().equals(gchair.getName())) {
								nbAuth = Math.max(1, nbAuth-1);
							}
							else if(!authList.contains(p)) {
								authList.add(p);
							}
						}
					}

					for (Person p: authList) {
						authors.add(p.getName());
					}
					paper.put("authors", authors);
					paper.put("id", ""+i+offsetPaper);

					//type 20% poster, 20% demo 
					String type = "regular";
					double typeDouble = rand.nextDouble();
					if (typeDouble<=0.2) {
						type = "poster";
					}
					else if (typeDouble<= 0.4) {
						type = "demo";
					}
					paper.put("type", type);
					// decision dépend du type :poster ou démo --> posterAcceptRate. Autre acceptRate
					String decision ="";
					boolean accepted;
					//regular accept rate
					if(type.equals("regular")) {
						double decisionD = rand.nextDouble();
						if(decisionD <= acceptRate) {
							accepted = true;
							paper.put("decision", "accept");
						}
						else {
							accepted = false;
							paper.put("decision", "reject");
						}						
					}
					// poster/demo accept rate
					else {
						double decisionD = rand.nextDouble();
						if(decisionD <= posterAcceptRate) {
							accepted = true;
							paper.put("decision", "accept");
						}
						else {
							accepted = false;
							paper.put("decision", "reject");
						}	
					}
					
					
					// topics
					JSONArray papTopics = new JSONArray();
					int nbPapTopics= rand.nextInt(5)+3;
					for(int j = 0; j <nbPapTopics; j++) {
						papTopics.add(topics.get(rand.nextInt(topics.size())));
					}
					paper.put("keywords", papTopics);

					// si papier accepté, choisir 1 auteur presenter
					if(accepted) {
						Person pres = authList.get(rand.nextInt(authList.size()));
						pres.attendsConference();
						paper.put("presenter", pres.getName());
					}

					
					//set de reviews par papier
					int nbReviews = rand.nextInt(6-minReview)+minReview;
					// entre min rev et 5 reviews
					JSONArray reviews= new JSONArray();
					for (int r = 0; r < nbReviews; r++) {
						JSONObject review = new JSONObject();
						// if paper accepted, >60%accept, >20%weak accept, >10%weak reject
						double revRat = rand.nextDouble();
						if(accepted) {
							if(revRat<=0.4) {
								review.put("rate", "accept");
							}
							else if (revRat <= 0.80) {
								review.put("rate", "weak accept");
							}
							else {
								review.put("rate", "weak reject");
							}
						}
						// sinon >40% reject, >20% weak reject, >10% weak accept
						else {
							if(revRat<=0.4) {
								review.put("rate", "reject");
							}
							else if (revRat <= 0.60) {
								review.put("rate", "weak reject");
							}
							else {
								review.put("rate", "weak accept");
							}
						}
						
						// 20% de review avec un external reviewer
						JSONArray reviewers=new JSONArray();
						Person rev = pcMembers.get(rand.nextInt(pcMembers.size()));
						while(authList.contains(rev)) {
							rev= pcMembers.get(rand.nextInt(pcMembers.size()));
						}
						reviewers.add(rev.getName());
						double extRev = rand.nextDouble();
						if(extRev <=0.2) {
							Person extRevP = persons.get(rand.nextInt(persons.size()));
							while(authList.contains(extRevP)|| pcMembers.contains(extRevP)
									||externalReviewers.contains(extRevP)) {
								extRevP= persons.get(rand.nextInt(persons.size()));
							}
							reviewers.add(extRevP.getName());
							externalReviewers.add(extRevP);
						}
						review.put("author", reviewers);
						review.put("id", "Review "+r);
						
						reviews.add(review);
					}	
					// 1 meta review
					JSONObject metareview = new JSONObject();
					JSONArray reviewers=new JSONArray();
					Person rev = pcMembers.get(rand.nextInt(pcMembers.size()));
					while(authList.contains(rev)) {
						rev= pcMembers.get(rand.nextInt(pcMembers.size()));
					}
					reviewers.add(rev.getName());
					metareview.put("author", reviewers);
					metareview.put("id", "Metareview");
					if(accepted) {
						metareview.put("rate", "accept");
					}
					else {
						metareview.put("rate", "reject");
					}
					reviews.add(metareview);
					
					paper.put("reviews", reviews);
					paperArray.add(paper);
				}//***end for each paper


				FileWriter writerPap = new FileWriter("data_files/conferences/conf"+nConf+"_papers.json");
				writerPap.write(paperArray.toJSONString().replaceAll("\\\\", ""));
				writerPap.flush();
				writerPap.close();
				conference.put("paper_file", "/home/thieblin/eclipse-workspace/Conference-dataset-population/data_files/conferences/conf"+nConf+"_papers.json");

				JSONArray personsArray= new JSONArray();
				for(Person p : persons) {
					// les autres personnes sont aléatoirement étudiants et/ou participant
					p.randomSelectStudent();
					p.randomSelectParticipant();
					personsArray.add(p.toJSONObject());
				}
				FileWriter writerPers = new FileWriter("data_files/conferences/conf"+nConf+"_people.json");
				writerPers.write(personsArray.toJSONString().replaceAll("\\\\", ""));
				writerPers.flush();
				writerPers.close();
				conference.put("people_file", "/home/thieblin/eclipse-workspace/Conference-dataset-population/data_files/conferences/conf"+nConf+"_people.json");

				FileWriter writer = new FileWriter("data_files/conferences/conf"+nConf+".json");
				writer.write(conference.toJSONString().replaceAll("\\\\", ""));
				writer.flush();
				writer.close();
				

			}//***********END FOR EACH CONF*********************

		} catch (NotAFolderException | SparqlQueryMalFormedException | SparqlEndpointUnreachableException | IncompleteSubstitutionException | IOException | ParseException e) {
			e.printStackTrace();
		}
	}



}
