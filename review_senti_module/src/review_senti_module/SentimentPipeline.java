package review_senti_module;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.Options;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.DefaultPaths;
import edu.stanford.nlp.pipeline.ParserAnnotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentUtils;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.TreePrint;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Generics;











import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A wrapper class which creates a suitable pipeline for the sentiment
 * model and processes raw text.
 *<br>
 * The main program has the following options: <br>
 * <code>-parserModel</code> Which parser model to use, defaults to englishPCFG.ser.gz <br>
 * <code>-sentimentModel</code> Which sentiment model to use, defaults to sentiment.ser.gz <br>
 * <code>-file</code> Which file to process. <br>
 * <code>-stdin</code> Read one line at a time from stdin. <br>
 * <code>-output</code> pennTrees: Output trees with scores at each binarized node.  vectors: Number tree nodes and print out the vectors.  Defaults to printing just the root. <br>
 *
 * @author John Bauer
 */


public class SentimentPipeline {
  private static final NumberFormat NF = new DecimalFormat("0.0000");

  static class PolarityProb {
		float polar[] = new float[5];
		
		PolarityProb(){
			
		}
  }
  
  static enum Output {
    PENNTREES, VECTORS, ROOT, PROBABILITIES
  }

  static enum Input {
    TEXT, TREES
  }

  /**
   * Sets the labels on the tree (except the leaves) to be the integer
   * value of the sentiment prediction.  Makes it easy to print out
   * with Tree.toString()
   */
  static void setSentimentLabels(Tree tree) {
    if (tree.isLeaf()) {
      return;
    }
     
    for (Tree child : tree.children()) {
      setSentimentLabels(child);
    }

    Label label = tree.label();
    if (!(label instanceof CoreLabel)) {
      throw new IllegalArgumentException("Required a tree with CoreLabels");
    }
    CoreLabel cl = (CoreLabel) label;
    cl.setValue(Integer.toString(RNNCoreAnnotations.getPredictedClass(tree)));
  }

  /**
   * Sets the labels on the tree to be the indices of the nodes.
   * Starts counting at the root and does a postorder traversal.
   */
  static int setIndexLabels(Tree tree, int index) {
    if (tree.isLeaf()) {
      return index;
    }

    tree.label().setValue(Integer.toString(index));
    index++;
    for (Tree child : tree.children()) {
      index = setIndexLabels(child, index);
    }
    return index;
  }

  /**
   * Outputs the vectors from the tree.  Counts the tree nodes the
   * same as setIndexLabels.
   */
  static int outputTreeVectors(PrintStream out, Tree tree, int index) {
    if (tree.isLeaf()) {
      return index;
    }

    //out.print("  " + index + ":");

    //out.println();
    index++;
    for (Tree child : tree.children()) {
      index = outputTreeVectors(out, child, index);
    }
    return index;
  }

  /**
   * Outputs the scores from the tree.  Counts the tree nodes the
   * same as setIndexLabels.
   */
  static int outputTreeScores(PrintStream out, Tree tree, int index) {
    if (tree.isLeaf()) {
      return index;
    }

    out.print("  " + index + ":");
    SimpleMatrix vector = RNNCoreAnnotations.getPredictions(tree);
    
    for (int i = 0; i < vector.getNumElements(); ++i) {
      out.print("  " + NF.format(vector.get(i)));
      
      //System.out.println(" "+Float.parseFloat(NF.format(vector.get(i))));
      
    }
    out.println();
    index++;
    for (Tree child : tree.children()) {
      index = outputTreeScores(out, child, index);
    }
    return index;
  }
  
  /*n*/
  static int revOutputTreeScores(PrintStream out, Tree tree, int index, PolarityProb c,int num) {
	    if (tree.isLeaf()) {
	      return index;
	    }
	    
	    //out.print("index : " + index + ":");
	    //System.out.println(num);
	    SimpleMatrix vector = RNNCoreAnnotations.getPredictions(tree);
	    
	    if(index == num){
	    	
		    for (int i = 0; i < vector.getNumElements(); ++i) {
		    //  out.print("index== num 일때 ");
		      c.polar[i] = Float.parseFloat(NF.format(vector.get(i)));
		    //  System.out.println(" "+c.polar[i]);
		      
		    }
	    }
	   // out.println();
	    index++;
	    for (Tree child : tree.children()) {
	      index = revOutputTreeScores(out, child, index, c,num);
	    }
	    return index;
	  }
  /*n*/
  static PolarityProb outputProbability(PrintStream out, CoreMap sentence, List<Output> outputFormats, int num){
	  Tree tree = sentence.get(SentimentCoreAnnotations.AnnotatedTree.class);
	  PolarityProb c= new PolarityProb();
	  
	  revOutputTreeScores(out, tree, 0, c, num);
	return c;
  }
  

  /**
   * Outputs a tree using the output style requested
   */
  static void outputTree(PrintStream out, CoreMap sentence, List<Output> outputFormats) {
    Tree tree = sentence.get(SentimentCoreAnnotations.AnnotatedTree.class);
    for (Output output : outputFormats) {
      switch (output) {
      case PENNTREES: {
        Tree copy = tree.deepCopy();
        setSentimentLabels(copy);
       // out.println(copy);
        break;
      }
      case VECTORS: {
        Tree copy = tree.deepCopy();
        setIndexLabels(copy, 0);
        //out.println(copy);
        outputTreeVectors(out, tree, 0);
        break;
      } 
      case ROOT: {
        //out.println("  " + sentence.get(SentimentCoreAnnotations.ClassName.class));
        break;
      }
      case PROBABILITIES: {
        Tree copy = tree.deepCopy();
        setIndexLabels(copy, 0);
        //out.println(copy);
        outputTreeScores(out, tree, 0);
        break;
      }
      default:
        throw new IllegalArgumentException("Unknown output format " + output);
      }
    }
  }
  static Tree outputLabelTree(PrintStream out, CoreMap sentence, List<Output> outputFormats) {

	    Tree tree = sentence.get(SentimentCoreAnnotations.AnnotatedTree.class);
	    
	    Tree copy = tree.deepCopy();
	    setSentimentLabels(copy);
	     
	    return copy;
	        
  }

  public static void help() {
    System.err.println("Known command line arguments:");
    System.err.println("  -sentimentModel <model>: Which model to use");
    System.err.println("  -parserModel <model>: Which parser to use");
    System.err.println("  -file <filename>: Which file to process");
    System.err.println("  -fileList <file>,<file>,...: Comma separated list of files to process.  Output goes to file.out");
    System.err.println("  -stdin: Process stdin instead of a file");
    System.err.println("  -input <format>: Which format to input, TEXT or TREES.  Will not process stdin as trees.  Trees need to be binarized");
    System.err.println("  -output <format>: Which format to output, PENNTREES, VECTOR, PROBABILITIES, or ROOT.  Multiple formats can be specified as a comma separated list.");
    System.err.println("  -filterUnknown: remove neutral and unknown trees from the input.  Only applies to TREES input");
  }

  /**
   * Reads an annotation from the given filename using the requested input.
   */
  public static Annotation getAnnotation(Input inputFormat, String filename, boolean filterUnknown) {
    switch (inputFormat) {
    case TEXT: {
      String text = IOUtils.slurpFileNoExceptions(filename);
      Annotation annotation = new Annotation(text);
      return annotation;
    }
    case TREES: {
      List<Tree> trees = SentimentUtils.readTreesWithGoldLabels(filename);
      if (filterUnknown) {
        trees = SentimentUtils.filterUnknownRoots(trees);
      }
      List<CoreMap> sentences = Generics.newArrayList();
      
      for (Tree tree : trees) {
        CoreMap sentence = new Annotation(Sentence.listToString(tree.yield()));
        sentence.set(TreeCoreAnnotations.BinarizedTreeAnnotation.class, tree);
        sentences.add(sentence);
      }
      Annotation annotation = new Annotation("");
      annotation.set(CoreAnnotations.SentencesAnnotation.class, sentences);
      return annotation;
    }
    default:
      throw new IllegalArgumentException("Unknown format " + inputFormat);
    }
  }

  public static void parsing_sentence(String textDoc, String outputFileName, StringBuffer sb, boolean threeClass) throws IOException{
	  
	  	String regex = "(<sentence)(.+?)(</sentence>)"; //<S> ~ </S> 
	  
		Pattern p = Pattern.compile(regex, Pattern.CASE_INSENSITIVE | Pattern.DOTALL | Pattern.MULTILINE);
		Matcher m = p.matcher(textDoc);
		
		ArrayList<String> adj_list = new ArrayList<String>();
		HashMap<String , Integer> adjMap = new HashMap<String , Integer>();
		HashMap<String , Integer> nnMap = new HashMap<String , Integer>();
		
		m.find();
		while(true){
			String g = m.group();	  // input sentence <S>~</S>
			 
		  	
			
			/* POS information */
			String regex2 = "(<word)(.+?)(word>)";
			Pattern p2 = Pattern.compile(regex2, Pattern.CASE_INSENSITIVE | Pattern.DOTALL | Pattern.MULTILINE);
			Matcher m2 = p2.matcher(g);
			int index = 0;
			while( m2.find()){
			
				
				String a = m2.group();
				if( a.contains("\"JJ")){
					a = a.replaceAll("<[^>]*>","");
					a = a.trim();
					//System.out.println("adj : "  +  a);
					adj_list.add(a);	
					adjMap.put(a, index);
				}
				else if( a.contains("\"NN")){
					a = a.replaceAll("<[^>]*>","");
					a = a.trim();
					nnMap.put(a, index);
				}
				index++;
				
			}
		
		
			String regex3 = "<[^>]*>"; //모든 tag 제거
			g = g.replaceAll(regex3, "");
			g = g.replaceAll("\n", " ");
			g = g.replaceAll("   ", " ");
			g = g.trim();
			
			//String st[] = g.split(" "); // <W c="tag">~</W> 각각 저장됨
			
			//System.out.println(st[0]);
			//System.out.println(g.length());

			int non = 0;
			non = treeBank(g , adj_list, adjMap, nnMap, outputFileName,sb, threeClass);

			
			if(m.find()){
				if(non == 0)
					sb.append(",");
			}
			else
				break;
			
			adj_list.clear();
			adjMap.clear();
			nnMap.clear();
		  	
		}
  }
  public static int treeBank(String s, ArrayList<String> adj_list,	HashMap<String , Integer> adjMap,	HashMap<String , Integer> nnMap, String outputFileName,StringBuffer sb,boolean threeClass ) throws IOException{
		
	    boolean stdin = false;
	   
	    
	    /*문장 test 용*/
    	//sb.append(s);
    	// ;
    	
	    List<Output> outputFormats = Arrays.asList(new Output[] { Output.PROBABILITIES});
	    Input inputFormat = Input.TEXT;


	    Properties props = new Properties();

	    if (stdin) {
	      props.setProperty("ssplit.eolonly", "true");
	    }
	    if (inputFormat == Input.TREES) {
	      props.setProperty("annotators", "sentiment");
	      props.setProperty("enforceRequirements", "false");
	    } else {
	      props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");
	    }

	    
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    
	    
	    	String line = s;
	        String result = "";
	        if (line.length() > 0) {
	         
	          Annotation annotation = pipeline.process(line);
	          
	          
	          int numSen = 0;
	          int adjNumOfFirst=0;
	          for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class))
	          {
	        	  if( numSen ==0){
	        		  for(int k =0 ; k < adj_list.size(); k++){
	        			  String adj = adj_list.get(k);
	        			  if(sentence.toString().contains(adj))
	        				  adjNumOfFirst++;
	        		  }
	        	  }
	        	  numSen++;
	          }
	          if(numSen > 1) //sentence 분리 안되서 2개의 문장으로 되서 에러나는 것들 ...
	        	  return 1;
	      	  
	          // dependency parser loading
	          LexicalizedParser parser = LexicalizedParser.getParserFromFile(DefaultPaths.DEFAULT_PARSER_MODEL, new Options());
	          
	          int check = 0;
	          for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
	        	 		
	    		Tree parseTree = parser.parse(sentence.toString());
	    		//System.out.println(printTree(parseTree, "typedDependencies"));
	    		System.out.println();
	    		System.out.println();
	    		System.out.println();
	    		System.out.println("input sentence : " + sentence.toString());
	    		 Tree t = outputLabelTree(System.out, sentence, outputFormats);
	    		 
	        	 result = t.toString();
	             System.out.println("result :: " + result);
	             //outputTree(System.out, sentence, outputFormats);
	        	 
	        	 if(check>0) //예외처리 . ? 가 문장으로 인싱되서 json 꺠짐 
	        		 break;
	        	 check++;
	                    	
	         	sb.append("{\"original sentence is\":"+"\""+sentence+"\",");  //original sentence 객체 시작
	         	
	         	sb.append("\"word\":["); //word 배열 시작
	        	 
	        	 //모든 형용사에 대하여.
	            int start = 0;
	            for(int i =0 ; i < adj_list.size(); i++){
	            	String adj = adj_list.get(i);
	            	
	            	//형용사의 확률 구함
	            	
	            	int endResult = result.indexOf(adj_list.get(i));
	            	int adj_index = 0;
	            	if( endResult != -1){
		            	String st = result.substring(0,	endResult ); //tree 출력상에서 형용사의 index를 계산함
			      
			            	for(int j=0; j<st.length();j++){
			            		if( st.charAt(j) == '(')
			            			adj_index++;	
			            	}
			            //	System.out.println( adj_list.get(i)  + "'s index is : "  +  adj_index);
	            	}
	            	
		            PolarityProb c = outputProbability(System.out, sentence, outputFormats, adj_index-1);			         
		            
	            	
	            	//System.out.println("search for adj : "  + adj);
	            	int sub_end = result.indexOf(adj, start);
	            	
	            	if(sub_end == -1){
	            		//System.out.println("failed search for " + adj);
	            	}
	            	else{
	            	String senti = result.substring(sub_end -2 , sub_end-1);
	            	
	            	int indexOfAdj = adjMap.get(adj);
	            	
	            	Set<Entry<String, Integer>> set = nnMap.entrySet();
	            	Iterator<Entry<String, Integer>> it = set.iterator();	
	            	
	            	
	            	/*가장 가까운 명사 찾기*/
	            	int min =1000;
	            	String noun ="";
	            	while (it.hasNext()) {
	        			Map.Entry<String, Integer> e = (Map.Entry<String, Integer>)it.next();      			
		            	
	        			int dis = findShortestNoun(t, indexOfAdj, e.getValue());

		            	if ( dis <= min){  //등호로 인해 같은 거리 일때 나중에 것을 선택하도록 됨
		            		noun = e.getKey();
		            		min = dis;
		            	}
		            		
		            	/*//그냥 거리
	        			if(  Math.abs(e.getValue()-indexOfAdj) < min){
	        				noun = e.getKey();
	        				min = Math.abs(e.getValue()-indexOfAdj);
	        			}*/
	            	}
	            	
	            	
	            	//문자 변환
	            	if(senti.contains("0"))
	            		senti = "--";
	            	else if(senti.contains("1"))
	            		senti = "-";
	            	else if(senti.contains("2"))
	            		senti = "neutral";
	            	else if(senti.contains("3"))
	            		senti = "+";
	            	else if(senti.contains("4"))
	            		senti = "++";
	            	
	            	/* 파일 출력*/   	
	            	if( isNetral(c)){
	            		
		            	System.out.println("pair : "  + adj + " , "+ noun);
	            		
	            		String pol = null;
		            	if(threeClass==true){
		            		pol= 	
		            				"\"positive\":"+"\""+Float.toString((c.polar[3]+c.polar[4]))+"\""+","+
		            				"\"neutral\":"+"\""+Float.toString((c.polar[2]))+"\""+","+
		            				"\"negative\":"+"\""+Float.toString((c.polar[0]+c.polar[1]))+"\"";				
		            	}	
		            	else{
		            		pol= 	"\"strong positive\":"+"\""+Float.toString((c.polar[4]))+"\""+","+
		            				"\"positive\":"+"\""+Float.toString((c.polar[3]))+"\""+","+
		            				"\"neutral\":"+"\""+Float.toString((c.polar[2]))+"\""+","+
		            				"\"negative\":"+"\""+Float.toString((c.polar[1]))+"\""+","+
		            				"\"strong negative\":"+"\""+Float.toString((c.polar[0]))+"\"";
		            	}
		            	
		            	String out = "{"+
		            				"\"opinion word\":"+"\""+adj+"\""+","+
		            				"\"opinion target\":"+"\""+noun+"\""+","+
		            				"\"polarity score\":"+"[{"+pol+"}]"+
		            				"}";
		            	
		            	//병렬되는 word:{},{} 를 위해 콤마 찍어줌
		            	if(i != (adj_list.size()-1) && i < adjNumOfFirst-1){ //numSen<2는 전처리 단계에서 sentence 분리가 안되서, 이곳에서 분리되어 오류를 만듬
		            		out = out+",";
		            	}
		            	sb.append(out);
		            	
	            	}
	            	
	            	
	            	//System.out.println("("+noun+","+adj+","+senti +")" );
	            	}
	            	
	            	start = sub_end;
	            	
	            	
	            }//adj for end
	            
	            sb.append("]"); //word 배열 끝
	            if((sb.substring(sb.length()-2)).contains(",]")){
	            	sb.deleteCharAt(sb.length()-2);
	            }
	            
	            
	          }//sentence for end
	          sb.append("}");//original sentence 객체 종료
	        
	        } else {
	          // Output blank lines for blank lines so the tool can be
	          // used for line-by-line text processing
	         // System.out.println("");
	        }
			return 0;
	      
	      
	    
	  }
  
   /*xml 결과 폴더 안에 .txt파일 모두 불러오기*/
	public static List<File> addTxtFile(String dirName) {
		File root = new File(dirName);
		List<File> files = new ArrayList<File>();
		for (File child : root.listFiles()) {
			if (child.isFile() && child.getName().endsWith(".txt")) {
				files.add(child); 
			} 
			
		}
		return files;
	}
	public static int findShortestNoun(Tree t , int adj, int candidate){
		
	    List<Tree> tr = t.getLeaves();
    	List<Tree> adjTreeList = new ArrayList<Tree>();
    	List<Tree> nounTreeList = new ArrayList<Tree>();
	    Tree a = tr.get(adj).parent(t); //change index to input variable
	    while( !a.equals(t) ){
	    	//insert data
	    	a = a.parent(t);
	    	adjTreeList.add(a);
	    }
	    a = tr.get(candidate).parent(t); //change index to input variable
	    while( !a.equals(t) ){
	    	//insert data
	    	a = a.parent(t);
	    	nounTreeList.add(a);
	    }
	    int distance = 100 ;
	    for(int i = 0 ; i < adjTreeList.size() ; i++){
	    	for( int j = 0 ; j < nounTreeList.size() ; j++){
	    		if(adjTreeList.get(i).equals(nounTreeList.get(j)) && (i+j) < distance){
	    			 distance = i + j;
	    		}
	    	}
	    }
	    return distance;
	}
	public static boolean isNetral(PolarityProb a){
		
		for( int i = 0 ; i < 5 ; i++){
			if( a.polar[2] < a.polar[i])
				return true;
		}
		
		return false;
		
	}
  
  public static void senti(String inputDirPath, String outputDirPath, Boolean threeClass)throws IOException, ClassNotFoundException{
	  List<File> files = addTxtFile(inputDirPath); // input directory
		

		for (int i = 0; i < files.size(); i++) {
			StringBuffer sb = new StringBuffer();
			File file = files.get(i);
		    String outputFileName = outputDirPath +"/"+ file.getName();
			BufferedWriter output = new BufferedWriter(new FileWriter(outputFileName)); //output directory	
			sb.append("{\"document\":[");  //document 배열 시작

			//System.out.println("File num is" + i);
			BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(file),"UTF-8"));
			
			//System.out.println(file.getCanonicalPath());
			String taggedDoc = Tagger_IRNLP.tagger(file.getCanonicalPath());   // file 의 path를 받아서 tagged string 리턴
			//System.out.println(taggedDoc);

			
			//if(i !=0 ) //file path 배열의 , 구분
				//sb.append(",");
			
			String fileName="\""+file.getName()+"\"";   //file path value
			sb.append("{\"file name\":"+fileName); //file path 객체 시작
		    sb.append(",\"sentence\":["); //sentence 배열 시작
		    

			parsing_sentence(taggedDoc, outputFileName, sb, threeClass);
			
			if((sb.substring(sb.length()-1)).contains(",")){
				//System.out.println(sb.substring(sb.length()-1));
				sb.deleteCharAt(sb.length()-1);  //마지막 문장 , 예외처리
			
			
			}
			
			sb.append("]"); //sentence 배열 끝
			input.close();
			sb.append("}"); //file path 객체 끝
			
			sb.append("]}"); //document 배열,객체 끝
			output.write(sb.toString());
			output.close();
			
		}
		System.out.println("The anlysis has been done.");
  
	  
  }
	
  public static void senti(String property)throws IOException, ClassNotFoundException{
	  
	  
	  String inputDirPath ="";
	  String outputDirPath ="";
	  Boolean threeClass = false;
	  
	  BufferedReader prop = new BufferedReader(new InputStreamReader(new FileInputStream(property),"UTF-8"));
	  inputDirPath = prop.readLine().toString().trim();
	  outputDirPath = prop.readLine().toString().trim(); 
	  
	  inputDirPath = inputDirPath.substring(1, inputDirPath.length());
	  

	  File outputD = new File(outputDirPath);
	  if(!outputD.exists()){
		  outputD.mkdir();	
	  }
		

	  if(prop.readLine().equalsIgnoreCase("Y"))
		  threeClass = true;
	  
	  prop.close();
	  
	  
	  
	  List<File> files = addTxtFile(inputDirPath); // input directory
		

		for (int i = 0; i < files.size(); i++) {
			StringBuffer sb = new StringBuffer();
			File file = files.get(i);
		    String outputFileName = outputDirPath +"/"+ file.getName();
		   
			BufferedWriter output = new BufferedWriter(new FileWriter(outputFileName)); //output directory	
			sb.append("{\"document\":[");  //document 배열 시작

			//System.out.println("File num is" + i);
			BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(file),"UTF-8"));
			
			//System.out.println(file.getCanonicalPath());
			String taggedDoc = Tagger_IRNLP.tagger(file.getCanonicalPath());   // file 의 path를 받아서 tagged string 리턴
			//System.out.println(taggedDoc);

			
			//if(i !=0 ) //file path 배열의 , 구분
				//sb.append(",");
			
			String fileName="\""+file.getName()+"\"";   //file path value
			sb.append("{\"file name\":"+fileName); //file path 객체 시작
		    sb.append(",\"sentence\":["); //sentence 배열 시작
		    

			parsing_sentence(taggedDoc, outputFileName, sb, threeClass);
			
			if((sb.substring(sb.length()-1)).contains(",")){
				//System.out.println(sb.substring(sb.length()-1));
				sb.deleteCharAt(sb.length()-1);  //마지막 문장 , 예외처리
			
			
			}
			
			sb.append("]"); //sentence 배열 끝
			input.close();
			sb.append("}"); //file path 객체 끝
			
			sb.append("]}"); //document 배열,객체 끝
			output.write(sb.toString());
			output.close();
			
		}
		System.out.println("The anlysis has been done.");
  
	  
  }
	public static String printTree(Tree tree, String format) {
		StringWriter sw = new StringWriter();
		PrintWriter pw = new PrintWriter(sw);
		TreePrint oTreePrint = new TreePrint(format);
		oTreePrint.printTree(tree, pw);
		return sw.toString().trim();
	}
  
  public static void main(String[] args) throws IOException, ClassNotFoundException {

	  boolean threeClass = false;
	  String outputDirPath = ""; 
	  String inputDirPath = "";

	  if(args.length < 1){
		  System.out.println("Invald format!\ndefine <property file> or <Input Directory Path> <Output Directory Path> (option)-threeClass");
		  System.exit(0);
	  } else if(args.length == 1) {
		  File p = new File(args[0]);
		  if(!p.exists()){
			  
			  System.out.println("Invald format!\ndefine <property file> or <Input Directory Path> <Output Directory Path> (option)-threeClass");
			  System.exit(0);
		  }
		  senti(args[0]);
	  } else {
		  for(int argIndex = 0; argIndex < args.length; argIndex++){
			  if(args[argIndex].equalsIgnoreCase("-threeClass")){
				  if(args.length==2){
					  System.out.println("Invald format!\njava -jar review_senti_module.jar <Input Directory Path> <Output Directory Path> (option)-threeClass");
					  System.exit(0);
				  }				
				  threeClass = true;
			  }			
		  }


		  inputDirPath = args[0];
		  outputDirPath = args[1];


		  File inputD = new File(inputDirPath);
		  File outputD = new File(outputDirPath);
		  if(	!inputD.exists()){
			  System.out.println("Wrong input directory!\njava -jar review_senti_module.jar <Input Directory Path> <Output Directory Path> (option)-threeClass");
			  System.exit(0);
		  }
		  if(	!outputD.exists()){
			  if(!outputD.mkdir()) {
				  System.out.println("Wrong output directory!\njava -jar review_senti_module.jar <Input Directory Path> <Output Directory Path> (option)-threeClass");
				  System.exit(0);
			  }		
		  }

		  senti(inputDirPath, outputDirPath, threeClass);
	  }
  }
}

