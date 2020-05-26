import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import parser.PageParser;
import terminal.Getopt;

public class AppExtraction {

	private static String __outputFile;
	private static List<File> __files;

	private static int nbErrors;
	private static List<String> errors;
	
	private static int nb = 0;

	

	public static void main(String[] args) {

		if (args.length < 1) {
			printUsage();
			System.exit(1);
		}

		__outputFile = "output.json";
		__files = new ArrayList<>();

		
		try {
			Getopt g = new Getopt(args, "o:");
			int c = -1;
			while ((c = g.getopt()) != -1) {
				switch (c) {
				case 'o':
					__outputFile = g.getOptarg();
					break;
				}
			}

			for (int i = g.getOptind(); i < args.length; i++)
				__files.add(new File(args[i]));

		} catch (Exception e) {
			printUsage();
			// e.printStackTrace();
			System.exit(1);
		}

		nbErrors = 0;
		errors = new ArrayList<>();
		
		try {

			JSONArray dataArray = new JSONArray();

			System.out.println("Parsing file(s) ...");

			for (File input : __files) {

				if (input.isDirectory()) {

					System.out.println("Scanning " + input.getName() + " directory");

					List<File> files = Arrays.asList(input.listFiles());
					files.forEach(inputFile -> {
						if (inputFile.getName().endsWith(".html")) {
							nb++;
							System.out.println("> " + inputFile.getName());
							JSONObject data = process(inputFile);
							dataArray.add(data);
						}
					});

				} else {
					System.out.println("> " + input.getName());
					JSONObject data = process(input);
					dataArray.add(data);
				}

				FileWriter writer = new FileWriter(__outputFile);
				writer.write(dataArray.toJSONString());
				writer.flush();
				writer.close();

			}

			System.out.println("\n\nFile(s) parsed !");
			System.out.println("\n\nnb files : "+ nb);		
			System.out.println("\n** parsing errors : " + nbErrors);
			if (nbErrors != 0) {
				errors.forEach(System.out::println);
				System.exit(1);
			}			
			
			System.out.println("Data output file : " + __outputFile);
		

		} catch (IOException e) {
			System.err.println("Error, can't open or write into file : " + __outputFile);
			e.printStackTrace();
		}

		System.out.println("\n*** END ***");

	}

	

	private static JSONObject process(File file) {
		try {
			return PageParser.parse(file);
		} catch (IOException e) {
			// System.err.println("Error parsing file : " +
			// file.getAbsolutePath());
			// e.printStackTrace();
			nbErrors++;
			errors.add("Error parsing file : " + file.getAbsolutePath());
		}
		return null;
	}

	private static void printUsage() {
		System.err.println("Usage : AppExtraction [options] input...");
		System.err.println("input -> input files or input directory.");
		System.err.println("Options : ");
		System.err.println("\t-o <file>\n\t\toutput file. (Optional) \"output.json\" if missing.");
	}

}
