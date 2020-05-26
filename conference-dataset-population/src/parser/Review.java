package parser;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.jsoup.nodes.Element;

public class Review {
	
	private String id;
	private JSONArray authors;
	private Element comment;
	
	public Review(String id, String author, Element comment) {
		this.id = id;
		 authors = new JSONArray();
		authors.add(author);
		this.comment = comment;
	}
	
	public Review(String id, String author, String commentText) {
		this.id = id;
		 authors = new JSONArray();
		authors.add(author);
		this.comment = new Element("comment").text(commentText);
	}
	
	public String getId() {
		return this.id;
	}
	
	public JSONArray getAuthor() {
		return this.authors;
	}
	
	public Element getComment() {
		return this.comment;
	}
	
	public JSONObject toJSON() {
		JSONObject obj = new JSONObject();
		obj.put("id", this.id);
		//todo: change into author table
		obj.put("author", this.authors);
		//obj.put("comment", this.comment.text());
		return obj;
	}
	
	public static String[] getReviewInfo(String info) {
		Pattern p;
		Matcher m;
		
		p = Pattern.compile("[Rr]eview ([1-9]+) [\\(]?by ([^\\)]+)[\\)]?");
		m = p.matcher(info);
		
		if(m.find())
			return new String[] {"Review " + m.group(1), m.group(2) };
		
		p = Pattern.compile("Metareview by (.+)");
		m = p.matcher(info);
		
		if(m.find())
			return new String[] {"Metareview", m.group(1)};
		
		return null;
	}

	
	@Override
	public String toString() {
		return this.id + " by " + this.authors + "\n" + this.comment;
	}

}
