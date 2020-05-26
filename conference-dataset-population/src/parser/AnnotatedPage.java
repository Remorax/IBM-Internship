package parser;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

/**
 * Representative class of an annotated HTML page
 * @author Thomas CANTIE
 * @see Page
 */
public class AnnotatedPage extends Page {

	/**
	 * Constructor
	 * 
	 * @param file
	 *            the HTML page to use
	 */
	public AnnotatedPage(File file) {
		super(file);
	}

	@Override
	public void parse() throws IOException {
		Document doc = Jsoup.parse(this.file, "UTF-8", "");

		// title
		this._title = doc.selectFirst("strong[property=dcterms:title]");

		// authors
		this._authors = doc.select("p[rel=dcterms:creator] > "
				+ "span[typeof=foaf:Person][property=foaf:name]");

		// abstract
		this._abstract = doc.selectFirst("span[property=dcterms:abstract]").text();

		// keywords
		this._keywords = doc.select("span[property=prism:keyword]");
		
		//type
		//System.out.println("hi"+doc.select("title").text());
		this.type = doc.select("title").text();
		this.type = this.type.substring(this.type.indexOf('(')+1,this.type.indexOf(')'));

		// decision
		this._decision = doc.selectFirst("strong:contains(Decision:) + span");

		// reviews
		Elements reviews = doc.select("div[typeof=fabio:Review]");

		this._reviews = new ArrayList<Review>();

		reviews.forEach(review -> {
			Element revId = review.selectFirst("strong[rel=dcterms:creator]");
			String[] revInfo = Review.getReviewInfo(revId.text());
			Element comment = review.selectFirst("pre[typeof=fabio:Comment]");
			this._reviews.add(new Review(revInfo[0], revInfo[1], comment));
		});
	
	}


}
