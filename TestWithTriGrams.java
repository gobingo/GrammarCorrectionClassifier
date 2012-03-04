import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.ngram.NGramModel;
import opennlp.tools.util.StringList;


public class TestWithTriGrams {

	private static ArrayList<WeightFeatureMap> WeightFeatureMapObjs_On = null;
	private static ArrayList<WeightFeatureMap> WeightFeatureMapObjs_In = null;
	private static ArrayList<WeightFeatureMap> WeightFeatureMapObjs_Of = null;
	private static int on_class_weight=0, in_class_weight=0, of_class_weight=0;
	private static float correctlyPredicted=0, wronglyPredicted=0, tpIn=0,tnIn=0,fpIn=0,fnIn=0,tpOn=0,tnOn=0,fpOn=0,fnOn=0,tpOf=0,tnOf=0,fpOf=0,fnOf=0;
	private static ArrayList<String> trainingTokens = null;
	
	private static ArrayList<ArrayList<TrigramElement>> dicList=null;
	private static ArrayList<ArrayList<TrigramElement>> devDicList=null;
	private static ArrayList<ArrayList<TrigramElement>> testDicList=null;
	private static Dictionary inTrigrams=null;
	private static Dictionary onTrigrams=null;
	private static Dictionary ofTrigrams=null;
	
	public static class TrigramElement{
		public StringList trigram;
		public String prep;
		public TrigramElement(StringList cTrigram, String cPrep){
			trigram=cTrigram;
			prep=cPrep;
		}
	}
	
	public static class WeightFeatureMap{
		public int weight;
		public StringList feature;
		public WeightFeatureMap(StringList cFeature, int cWeight){
			weight=cWeight;
			feature=cFeature;
		}
		public int getWeight() {
			return weight;
		}
		public void setWeight(int weight) {
			this.weight = weight;
		}
		public StringList getFeature() {
			return feature;
		}
		public void setFeature(StringList feature) {
			this.feature = feature;
		}
		
	}
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try{
			System.out.println("Start Test!!");
	
			trainingTokens=new ArrayList<String>();
			File training_set_dir = new File("new_training_set");
			File development_set_dir = new File("new_development_set");
			File test_set_dir = new File("new_test_set");
			File[] training_set_files = training_set_dir.listFiles();
			File[] development_set_files = development_set_dir.listFiles();
			File[] test_set_files = test_set_dir.listFiles();
			
			InputStream posModelStream = new FileInputStream("en-pos-maxent.bin");
			POSModel posModel = new POSModel(posModelStream);
			InputStream tkModelStream = new FileInputStream("en-token.bin");
			TokenizerModel tkModel = new TokenizerModel(tkModelStream);
			InputStream sentenceModelStream = new FileInputStream("en-sent.bin");
			SentenceModel sentModel = new SentenceModel(sentenceModelStream);
			SentenceDetectorME sentDetector = new SentenceDetectorME(sentModel);
			
			devDicList=new ArrayList<ArrayList<TrigramElement>>();
			testDicList=new ArrayList<ArrayList<TrigramElement>>();
			
			initializeDataSet(development_set_files,sentDetector,posModel,tkModel,devDicList);
			initializeDataSet(test_set_files,sentDetector,posModel,tkModel,testDicList);
			
			generateTrigramFeatureVectors(training_set_files,sentDetector,posModel,tkModel);
			
			WeightFeatureMapObjs_In = initializeWeightVector(inTrigrams,WeightFeatureMapObjs_In,"IN");
			WeightFeatureMapObjs_On = initializeWeightVector(onTrigrams,WeightFeatureMapObjs_On,"ON");
			WeightFeatureMapObjs_Of = initializeWeightVector(ofTrigrams,WeightFeatureMapObjs_Of,"OF");
			
			train(dicList,WeightFeatureMapObjs_In,WeightFeatureMapObjs_On,WeightFeatureMapObjs_Of);
			System.out.println("With training data - Num correctly predicted: "+correctlyPredicted);
			System.out.println("With training data - Num wrongly predicted: "+wronglyPredicted);
			computeAndPrintPerformance(tpIn,tnIn,fpIn,fnIn,"In");
			computeAndPrintPerformance(tpOn,tnOn,fpOn,fnOn,"On");
			computeAndPrintPerformance(tpOf,tnOf,fpOf,fnOf,"Of");
			System.out.println();
			
			resetDefaults();
			train(devDicList,WeightFeatureMapObjs_In,WeightFeatureMapObjs_On,WeightFeatureMapObjs_Of);
			System.out.println("With dev data - Num correctly predicted: "+correctlyPredicted);
			System.out.println("With dev data - Num wrongly predicted: "+wronglyPredicted);
			computeAndPrintPerformance(tpIn,tnIn,fpIn,fnIn,"In");
			computeAndPrintPerformance(tpOn,tnOn,fpOn,fnOn,"On");
			computeAndPrintPerformance(tpOf,tnOf,fpOf,fnOf,"Of");
			System.out.println();
			
			resetDefaults();
			train(testDicList,WeightFeatureMapObjs_In,WeightFeatureMapObjs_On,WeightFeatureMapObjs_Of);
			System.out.println("With test data - Num correctly predicted: "+correctlyPredicted);
			System.out.println("With test data - Num wrongly predicted: "+wronglyPredicted);
			computeAndPrintPerformance(tpIn,tnIn,fpIn,fnIn,"In");
			computeAndPrintPerformance(tpOn,tnOn,fpOn,fnOn,"On");
			computeAndPrintPerformance(tpOf,tnOf,fpOf,fnOf,"Of");
			System.out.println();

			System.out.println("End Test!!");
			
		}catch(Exception ex){
			ex.printStackTrace();
		}
	}
	
	private static void resetDefaults() {
		correctlyPredicted=0;wronglyPredicted=0;tpIn=0;tnIn=0;fpIn=0;fnIn=0;tpOn=0;tnOn=0;fpOn=0;fnOn=0;tpOf=0;tnOf=0;fpOf=0;fnOf=0;
	}
	
    /**
     * Computes and prints classification performance
     *
     * @param tp: number of true positives
     * @param tn: number of true negatives
     * @param fp: number of false positives
     * @param fn: number of false negatives
     */
	private static void computeAndPrintPerformance(float tp,float tn,float fp,float fn,String type){
		float accuracy=((tp+tn)*100)/(tp+tn+fp+fn);
		float precision=(tp*100)/(tp+fp);
		float recall=(tp*100)/(tp+fn);
		float f_score=(2*precision*recall)/(precision+recall);
		System.out.println("Num of true positves= "+tp);
		System.out.println("Num of true negatives= "+tn);
		System.out.println("Num of false positves= "+fp);
		System.out.println("Num of false negatives= "+fn);
		System.out.println("Accuracy"+type+"= "+accuracy);
		System.out.println("Precision"+type+"= "+precision);
		System.out.println("Recall"+type+"= "+recall);
		System.out.println("F_Score"+type+"= "+f_score);
	}
	
	
    /**
     * Generates trigram features over training data
     *
     * @param input data set
     * @param sentence detector for the input data files
     * @param parts of speech tag model to detect parts of speech in the input data files
     * @param tokenizer model to retrieve tokens from input data files
     */
	private static void generateTrigramFeatureVectors(File[] files, SentenceDetectorME sentDetector, POSModel posModel, TokenizerModel tkModel){
		try{
			NGramModel ngModel = new NGramModel();
			POSTaggerME tagger = new POSTaggerME(posModel);
			dicList=new ArrayList<ArrayList<TrigramElement>>();
			for(int fileIter=0; fileIter<files.length; fileIter++){
				FileInputStream fileStream = new FileInputStream(files[fileIter]);
				BufferedReader bfr = new BufferedReader(new InputStreamReader(fileStream));
				String rText="";
				//System.out.println("File #"+fileIter);
				while((rText = bfr.readLine())!=null){
					String[] sentences = sentDetector.sentDetect(rText);
					//System.out.println("Num sentences: "+sentences.length);
					ArrayList<TrigramElement> tempTGEList=null;
					for(String sentence:sentences){
						Tokenizer tokenizer = new TokenizerME(tkModel);
						String[] tokens = tokenizer.tokenize(sentence);
						String[] targetTokens = generateTargetTokens(tokens);
						
						tempTGEList=new ArrayList<TrigramElement>();
						int numSplits = targetTokens.length/3;
						
						for(int numSplitIter=0; numSplitIter<numSplits; numSplitIter++){
							String[] triplets = generateTriplets(numSplitIter,targetTokens);
							ArrayList<String> tmpList=new ArrayList<String>();
							tmpList.addAll(Arrays.asList(triplets));
							String[] tags = tagger.tag(triplets);
							TrigramElement tgElem=null;
							if(tmpList.contains("in")){
								tgElem = new TrigramElement(new StringList(tags), "in");
							}else if(tmpList.contains("on")){
								tgElem = new TrigramElement(new StringList(tags), "on");
							}else if(tmpList.contains("of")){
								tgElem = new TrigramElement(new StringList(tags), "of");
							}
							tempTGEList.add(tgElem);
						}
						
						for(String token:tokens){
							trainingTokens.add(token);
						}
						if(tempTGEList.size()>0)
							dicList.add(tempTGEList);
					}
					
				}
			}
			
			System.out.println("dicList created!");
			
			StringList sList = new StringList(trainingTokens.toArray(new String[trainingTokens.size()]));
			ngModel.add(sList,3,3);
			Dictionary trigrams = ngModel.toDictionary();
			inTrigrams=new Dictionary();
			onTrigrams=new Dictionary();
			ofTrigrams=new Dictionary();
			inTrigrams.put(new StringList(new String[]{"BIAS"}));
			onTrigrams.put(new StringList(new String[]{"BIAS"}));
			ofTrigrams.put(new StringList(new String[]{"BIAS"}));
			for(Iterator iter=trigrams.iterator();iter.hasNext();){
				StringList trigram = (StringList)iter.next();
				ArrayList<String> tempList=new ArrayList<String>();
				for(Iterator trigramIter=trigram.iterator();trigramIter.hasNext();){
					String trigramToken = (String) trigramIter.next();
					tempList.add(trigramToken);
				}
				if(tempList.contains("in")){
					String[] tags = tagger.tag(tempList.toArray(new String[tempList.size()]));
					inTrigrams.put(new StringList(tags));
				}
				if(tempList.contains("on")){
					String[] tags = tagger.tag(tempList.toArray(new String[tempList.size()]));
					onTrigrams.put(new StringList(tags));
				}
				if(tempList.contains("of")){
					String[] tags = tagger.tag(tempList.toArray(new String[tempList.size()]));
					ofTrigrams.put(new StringList(tags));
				}
			}
			
			System.out.println("trigram feature vectors created!");
		
		}catch(IOException ex){
			ex.printStackTrace();
		}
	}
	
    /**
     * Initializes dictionary list for input data set
     *
     * @param input data set
     * @param sentence detector for the input data files
     * @param parts of speech tag model to detect parts of speech in the input data files
     * @param tokenizer model to retrieve tokens from input data files
     * @param dictionary list to be populated
     */
	private static void initializeDataSet(File[] data_set_files,
			SentenceDetectorME sentDetector, POSModel posModel,
			TokenizerModel tkModel, ArrayList<ArrayList<TrigramElement>> dicList){
		try{
			POSTaggerME tagger = new POSTaggerME(posModel);
			for(int fileIter=0; fileIter<data_set_files.length; fileIter++){
				FileInputStream fileStream = new FileInputStream(data_set_files[fileIter]);
				BufferedReader bfr = new BufferedReader(new InputStreamReader(fileStream));
				String rText="";
				//System.out.println("File #"+fileIter);
				while((rText = bfr.readLine())!=null){
					String[] sentences = sentDetector.sentDetect(rText);
					//System.out.println("Num sentences: "+sentences.length);
					ArrayList<TrigramElement> tempTGEList=null;
					for(String sentence:sentences){
						Tokenizer tokenizer = new TokenizerME(tkModel);
						String[] tokens = tokenizer.tokenize(sentence);
						String[] targetTokens = generateTargetTokens(tokens);
						
						tempTGEList=new ArrayList<TrigramElement>();
						int numSplits = targetTokens.length/3;
						
						for(int numSplitIter=0; numSplitIter<numSplits; numSplitIter++){
							String[] triplets = generateTriplets(numSplitIter,targetTokens);
							ArrayList<String> tmpList=new ArrayList<String>();
							tmpList.addAll(Arrays.asList(triplets));
							String[] tags = tagger.tag(triplets);
							TrigramElement tgElem=null;
							if(tmpList.contains("in")){
								tgElem = new TrigramElement(new StringList(tags), "in");
							}else if(tmpList.contains("on")){
								tgElem = new TrigramElement(new StringList(tags), "on");
							}else if(tmpList.contains("of")){
								tgElem = new TrigramElement(new StringList(tags), "of");
							}
							tempTGEList.add(tgElem);
						}
						
						if(tempTGEList.size()>0)
							dicList.add(tempTGEList);
					}
					
				}
			}
			
			System.out.println("dicList created!");
		
		}catch(IOException ex){
			ex.printStackTrace();
		}
	}


	
	private static String[] generateTriplets(int numSplitIter,
			String[] targetTokens) {

		int loopCount=numSplitIter*3;
		ArrayList<String> triplets = new ArrayList<String>();
		for(int iter=loopCount;iter<(loopCount+3);iter++){
			triplets.add(targetTokens[iter]);
		}
		return triplets.toArray(new String[triplets.size()]);
	}
	

	private static String[] generateTargetTokens(String[] tokens) {

		ArrayList<String> targetTokens = new ArrayList<String>();
		for(int iter=0; iter<tokens.length; iter++){
			if(tokens[iter].equalsIgnoreCase("in") || tokens[iter].equalsIgnoreCase("on") || tokens[iter].equalsIgnoreCase("of")){
				if(iter>0 && tokens[iter-1]!=null) targetTokens.add(tokens[iter-1]);
				targetTokens.add(tokens[iter]);
				if(tokens[iter+1]!=null) targetTokens.add(tokens[iter+1]);
			}
		}
		return targetTokens.toArray(new String[targetTokens.size()]);
	}
	
    /**
     * 3-class perceptron logic for preposition selection
     *
     * @param input dictionary list
     * @param weight vectors for IN preposition
     * @param weight vectors for ON preposition
     * @param weight vectors for OF preposition
     */
	private static void train(ArrayList<ArrayList<TrigramElement>> dicList,
			ArrayList<WeightFeatureMap> weightVectorsIn,
			ArrayList<WeightFeatureMap> weightVectorsOn,
			ArrayList<WeightFeatureMap> weightVectorsOf) {
		
		String truePrep=null;
		String prepClassified=null;
		
		for(ArrayList<TrigramElement> dic:dicList){
			for(TrigramElement tgElem:dic){
				truePrep=tgElem.prep;
				in_class_weight=0;on_class_weight=0;of_class_weight=0;
				
				computeClassifierWeights(tgElem,weightVectorsIn,weightVectorsOn,weightVectorsOf);
				
				//printClassifierWeights(weightVectorsIn,weightVectorsOn,weightVectorsOf);
				
				if(in_class_weight>=on_class_weight && in_class_weight>of_class_weight){
					prepClassified = "in";
				}else if(on_class_weight>in_class_weight && on_class_weight>of_class_weight){
					prepClassified = "on";
				}else if(of_class_weight>=in_class_weight && of_class_weight>=on_class_weight){
					prepClassified = "of";
				}
				
				if(prepClassified.equals(truePrep)){
					//System.out.println("Classified correctly!!");
					if(prepClassified.equals("in")){
						tpIn++;
					}else if(prepClassified.equals("on")){
						tpOn++;
					}else if(prepClassified.equals("of")){
						tpOf++;
					}
					correctlyPredicted++;
				}else{
					//System.out.println("Classified wrongly!!");
					if(truePrep.equals("in") && !prepClassified.equals("on")){
						tnOn++;
					}
					if(truePrep.equals("in") && !prepClassified.equals("of")){
						tnOf++;
					}
					
					if(truePrep.equals("on") && !prepClassified.equals("in")){
						tnIn++;
					}
					if(truePrep.equals("on") && !prepClassified.equals("of")){
						tnOf++;
					}
					
					if(truePrep.equals("of") && !prepClassified.equals("on")){
						tnOn++;
					}
					if(truePrep.equals("of") && !prepClassified.equals("in")){
						tnIn++;
					}
					
					if(!truePrep.equals("in") && prepClassified.equals("in")){
						fpIn++;
					}
					
					if(!truePrep.equals("on") && prepClassified.equals("on")){
						fpOn++;
					}
					
					if(!truePrep.equals("of") && prepClassified.equals("of")){
						fpOf++;
					}
					
					if(truePrep.equals("in") && !prepClassified.equals("in")){
						fnIn++;
					}
					
					if(truePrep.equals("on") && !prepClassified.equals("on")){
						fnOn++;
					}
					
					if(truePrep.equals("of") && !prepClassified.equals("of")){
						fnOf++;
					}
					
					wronglyPredicted++;
					updateWeightsNew(tgElem,weightVectorsIn,weightVectorsOn,weightVectorsOf,truePrep,prepClassified);
					
					//printClassifierWeights(weightVectorsIn,weightVectorsOn,weightVectorsOf);
				}
			}

		}
		
	}
	
	private static void updateWeightsNew(TrigramElement tgElem,
			ArrayList<WeightFeatureMap> weightVectorsIn,
			ArrayList<WeightFeatureMap> weightVectorsOn,
			ArrayList<WeightFeatureMap> weightVectorsOf, String truePrep,
			String prepClassified) {

		if(truePrep.equals("in")){
			increaseWeights(weightVectorsIn,tgElem);
		}
		if(truePrep.equals("on")){
			increaseWeights(weightVectorsOn,tgElem);
		}
		if(truePrep.equals("of")){
			increaseWeights(weightVectorsOf,tgElem);
		}
		
		if(prepClassified.equals("in")){
			decreaseWeights(weightVectorsIn,tgElem);
		}
		if(prepClassified.equals("on")){
			decreaseWeights(weightVectorsOn,tgElem);
		}
		if(prepClassified.equals("of")){
			decreaseWeights(weightVectorsOf,tgElem);
		}
	}

	private static void decreaseWeights(
			ArrayList<WeightFeatureMap> weightVectors, TrigramElement tgElem) {

		for(WeightFeatureMap wfmObj:weightVectors){
			if(wfmObj.feature.compareToIgnoreCase(tgElem.trigram) || wfmObj.feature.getToken(0).equals("BIAS")){
				wfmObj.setWeight(wfmObj.weight-1);
			}
		}
	}

	private static void increaseWeights(
			ArrayList<WeightFeatureMap> weightVectors, TrigramElement tgElem) {

		for(WeightFeatureMap wfmObj:weightVectors){
			if(wfmObj.feature.compareToIgnoreCase(tgElem.trigram) || wfmObj.feature.getToken(0).equals("BIAS")){
				wfmObj.setWeight(wfmObj.weight+1);
			}
		}
	}

	private static void computeClassifierWeights(
			TrigramElement tgElem,
			ArrayList<WeightFeatureMap> weightVectorsIn,
			ArrayList<WeightFeatureMap> weightVectorsOn,
			ArrayList<WeightFeatureMap> weightVectorsOf) {
		
		for(WeightFeatureMap wfmObj:weightVectorsIn){
			if(wfmObj.feature.getToken(0).equals("BIAS")){
				in_class_weight=in_class_weight+(wfmObj.weight);
			}
			else if(wfmObj.feature.compareToIgnoreCase(tgElem.trigram)){
				in_class_weight=in_class_weight+(wfmObj.weight*1);
			}
		}
		
		for(WeightFeatureMap wfmObj:weightVectorsOn){
			if(wfmObj.feature.getToken(0).equals("BIAS")){
				on_class_weight=on_class_weight+(wfmObj.weight);
			}
			else if(wfmObj.feature.compareToIgnoreCase(tgElem.trigram)){
				on_class_weight=on_class_weight+(wfmObj.weight*1);
			}
		}
		
		for(WeightFeatureMap wfmObj:weightVectorsOf){
			if(wfmObj.feature.getToken(0).equals("BIAS")){
				of_class_weight=of_class_weight+(wfmObj.weight);
			}
			else if(wfmObj.feature.compareToIgnoreCase(tgElem.trigram)){
				of_class_weight=of_class_weight+(wfmObj.weight*1);
			}
		}
		
	}

    /**
     * Initializes weights for input trigrams based on classification type
     *
     * @return weight vector for corresponding preposition
     */
	private static ArrayList<WeightFeatureMap> initializeWeightVector(Dictionary trigrams,
			ArrayList<WeightFeatureMap> weightFeatureMapObjs, String classificationType) {
		
		weightFeatureMapObjs=new ArrayList<WeightFeatureMap>();
		boolean containsBias=false;
		for(StringList feature:trigrams){
			for(Iterator iter=feature.iterator();iter.hasNext();){
				String trigramElem=(String) iter.next();
				if(trigramElem.equals("BIAS")){
					containsBias=true;
					break;
				}
			}
			if(containsBias && classificationType.equals("OF")){
				WeightFeatureMap wfObj=new WeightFeatureMap(feature, 1);
				weightFeatureMapObjs.add(wfObj);
			}
			else{
				WeightFeatureMap wfObj=new WeightFeatureMap(feature, 0);
				weightFeatureMapObjs.add(wfObj);
			}
			containsBias=false;
		}
		
		return weightFeatureMapObjs;
		
	}

}