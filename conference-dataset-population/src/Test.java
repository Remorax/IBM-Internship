import java.io.FileReader;
import java.io.IOException;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class Test {

	public static void main(String[] args) {

		JSONParser parser = new JSONParser();

		try {

			Object object = parser.parse(new FileReader("./param.json"));

			JSONObject obj = (JSONObject) object;

			// System.out.println(object);

			System.out.println(obj.get("data-file"));

			((JSONArray) obj.get("params")).forEach(param -> {
				
				JSONObject entry = (JSONObject)  param;
					
				System.out.println(entry.get("name"));
				System.out.println(entry.get("endpoint"));
				System.out.println(entry.get("mapping"));
				System.out.println();
			});

		} catch (IOException e) {
			e.printStackTrace();
		} catch (ParseException e) {
			e.printStackTrace();
		}

	}

}
