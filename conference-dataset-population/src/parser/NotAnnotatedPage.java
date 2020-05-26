package parser;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

/**
 * Representative class of a non annotated HTML page
 * 
 * @author Thomas CANTIE
 * @see Page
 *
 */
public class NotAnnotatedPage extends Page {

	/**
	 * Constructor
	 * 
	 * @param file
	 *            the HTML page to use
	 */
	public NotAnnotatedPage(File file) {
		super(file);
	}

	@Override
	public void parse() throws IOException {
		Document doc = Jsoup.parse(file, "UTF-8", "");

		Element _entry = doc.selectFirst("div.entry-content");

		// title
		this._title = _entry.selectFirst("strong");

		// authors
		this._authors = _entry.select("strong:contains(Author(s):)").nextAll();

		// abstract
		this._abstract = _entry.selectFirst("strong:contains(Abstract:)")
				.siblingNodes().get(0).toString();

		// keywords
		this._keywords = _entry.select("strong:contains(Keywords:)").nextAll();
		
		this.type = doc.select("title").text();
		this.type = this.type.substring(this.type.indexOf('(')+1,this.type.indexOf(')'));

		// decision
		this._decision = _entry.selectFirst("strong:contains(Decision:) + span");

		// reviews
		this._reviews = new ArrayList<Review>();
		Elements revId = _entry.select("p strong:matches((?i)review)");		
		Elements revComments = _entry.select("p + pre");		
		
		for (int i = 0; i < revId.size(); i++) {
			String[] revInfo = Review.getReviewInfo(revId.get(i).text());
			_reviews.add(new Review(revInfo[0], revInfo[1], revComments.get(i)));
		}

	}
	
}
