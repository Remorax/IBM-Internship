package parser;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

/**
 * Representative class of an HTML page to parse
 * 
 * @author Thomas Canti�
 *
 */
public abstract class Page {

	/** file linked to the HTL page */
	protected File file;
	
	/* title of the publication */
	protected Element _title;

	/* authors of the publication */
	protected Elements _authors;
	
	/* abstract of the publication */
	protected String _abstract;

	/* keywords of the publication */
	protected Elements _keywords;

	/* decision for the publication */
	protected Element _decision;

	/* reviews of the publication */
	protected List<Review> _reviews;
	
	protected String type;

	/**
	 * Constructor
	 * 
	 * @param file
	 *            the HTML page to use
	 */
	public Page(File file) {
		this.file = file;
	}

	/**
	 * Parsing of the HTML page
	 * 
	 * @throws IOException
	 *             if the file doesn't exist or an IO error occurs
	 */
	public abstract void parse() throws IOException;

	/**
	 * Getter on extracted data
	 * 
	 * @return JSON-formated data
	 */
	public JSONObject getExtractedData() {
		JSONObject obj = new JSONObject();

		// name
		obj.put("id", this.file.getName());
		// title
		obj.put("title", this._title.text());
		// authors
		JSONArray arrayAuthors = new JSONArray();
		this._authors.forEach(author -> arrayAuthors.add(author.text()));
		obj.put("authors", arrayAuthors);
		if (this._decision.text().equals("accept")) {
			System.out.println("ALALA");
			obj.put("presenter", arrayAuthors.get((int) Math.random()*arrayAuthors.size()));
		}
		
		obj.put("type", this.type);
		// abstract
		//obj.put("abstract", this._abstract);
		// keywords
		JSONArray arrayKeywords = new JSONArray();
		this._keywords.forEach(keyword -> arrayKeywords.add(keyword.text()));
		obj.put("keywords", arrayKeywords);
		// decision
		obj.put("decision", this._decision.text());
		// reviews
		JSONArray array = new JSONArray();
		this._reviews.forEach(review -> {
			array.add(review.toJSON());
		});
		obj.put("reviews", array);
		//System.out.println("reviews: " + this._reviews.size());
		/*
		try {
			obj.put("hash", getHash(this._title.text()));
		} catch (NoSuchAlgorithmException e) {
			e.printStackTrace();
			obj.put("hash", "error : " + e.getMessage());
		}
		*/
		return obj;
	}

	/**
	 * Prints extracted data
	 */
	public void printBrutExtractedData() {
		System.out.println("*** " + this.file.getName());

		System.out.println("- Title : \n\t" + this._title.text());
		System.out.println("- Authors :");
		_authors.forEach(author -> System.out.println("\t" + author.text()));
		System.out.println("- Abstract : \n\t" + this._abstract);
		System.out.println("- Keywords :");
		_keywords.forEach(keyword -> System.out.println("\t" + keyword.text()));

		System.out.println("\n- Decision : \n\t" + this._decision.text());

		System.out.println("\n- Reviews :");
		System.out.println(this._reviews);
		/*
		try {
			System.out.println("-- hash : " + getHash(this._title.text()));
		} catch (NoSuchAlgorithmException e) {
			e.printStackTrace();
		}
		*/
	}

}
