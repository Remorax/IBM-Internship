package parser;

import java.io.File;
import java.io.IOException;

import org.json.simple.JSONObject;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

/**
 * Representative class of the parser
 * 
 * @author Thomas Canti�
 * @see Page
 * @see AnnotatedPage
 * @see NotAnnotatedPage
 *
 */
public class PageParser {

	/**
	 * Processing method
	 * 
	 * @param file
	 *            the HTML page to parse
	 * @return JSON-formated extracted data
	 * @throws IOException
	 *             if the file doesn't exist or an IO error occurs
	 */
	public static JSONObject parse(File file) throws IOException {

		Document doc = Jsoup.parse(file, "UTF-8", "");

		Page page;

		if (doc.select("div[rel=rdf:type] strong[property=dcterms:title]").size() == 1) {
			page = new AnnotatedPage(file);
		} else {
			page = new NotAnnotatedPage(file);
		}

		page.parse();
		System.out.println("reviews: " + page._reviews.size());
		return page.getExtractedData();

	}	

}
