package review_senti_module;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.StringTokenizer;


import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.tagger.maxent.TaggerConfig;

public class Tagger_IRNLP {
	public static String tagger(String inputFile) throws IOException,
			ClassNotFoundException {

		String model = "taggers/english-left3words-distsim.tagger";
		TaggerConfig config = new TaggerConfig("-outputFormat", "xml",
				"-model", model);

		MaxentTagger tagger = new MaxentTagger(model, config);
		String raw = TxtReader.readFile(inputFile);  // 각각의 file path

		String tagged = tagger.tagString(raw);
		//System.out.println(tagged);
		
		return tagged;

	}

	public static void writeFile(String outputString, String outputFileName) {
		try {

			BufferedWriter out = new BufferedWriter(new FileWriter(
					outputFileName));
			out.write(outputString);
			out.newLine();
			out.close();

		} catch (IOException e) {
			System.err.println(e);
			System.exit(1);
		}
	}


	public static HashMap<Integer, String[]> stringToMap(String input) {
		String[] token1 = deli(input, " ");
		HashMap<Integer, String[]> map = new HashMap<Integer, String[]>();

		for (int i = 0; i < token1.length; i++) {
			map.put(i, deli(token1[i], "_"));
		}

		return map;
	}

	public static String[] deli(String f_name, String de) {
		StringTokenizer st = new StringTokenizer(f_name, de);
		String arr[] = new String[st.countTokens()];
		int i = 0;
		while (st.hasMoreTokens()) {
			arr[i] = st.nextToken();
			i++;
		}
		return arr;

	}

}
