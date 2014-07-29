package review_senti_module;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.List;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.Options;
import edu.stanford.nlp.pipeline.DefaultPaths;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreePrint;

public class DependencyParsing {

	public static void main(String[] args) throws Exception {
		String sentence = "John loves the ice cream cake on my table.";
		LexicalizedParser parser = LexicalizedParser.getParserFromFile(DefaultPaths.DEFAULT_PARSER_MODEL, new Options());
		System.out.println("dfd");
		
		Tree parseTree = parser.parse(sentence);
		System.out.println(printTree(parseTree, "typedDependencies"));
	}
	
	public static String printTree(Tree tree, String format) {
		StringWriter sw = new StringWriter();
		PrintWriter pw = new PrintWriter(sw);
		TreePrint oTreePrint = new TreePrint(format);
		oTreePrint.printTree(tree, pw);
		return sw.toString().trim();
	}
}
