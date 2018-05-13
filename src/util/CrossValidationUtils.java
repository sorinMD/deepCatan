package util;

import java.util.ArrayList;

import org.deeplearning4j.eval.CatanPlotter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import data.CatanDataSet;
import data.CatanDataSetIterator;
import data.PreloadedCatanDataSetIterator;

/**
 * Simple utility to read and store the data, then split into train/validation
 * according to number of folds It also contains a few methods for writing the
 * results by averaging across folds and computing the standard deviation and
 * the standard error.
 * 
 * @author sorinMD
 */
public class CrossValidationUtils {
	/**
	 * The iterator over the full data
	 */
	private CatanDataSetIterator iter;
	private boolean init = false;
    
	/**
	 * Stores all the data that can then be accessed based on the index
	 */
	private ArrayList<CatanDataSet> fullData;
	private ArrayList<CatanDataSet> trainData;
	private ArrayList<CatanDataSet> testData;
	
	/**
     * nSamples is the number of samples we want from the data, but that may be greater then the actual size of the data
     */
    private int nSamples;
    /**
     * Keeps track of the actual amount of data used by the class, which is the minimum between the nSamples and the number of lines in the file
     */
    private int dataSamples;
    private int nFolds = 0;
    private int foldSize;
    private int foldStartIdx;
    private int foldEndIdx;
	
	/**
	 * Initialises all the parameters required for generating the train/validation data for each fold;
	 * Note: it doesn't create the first data set; to do this, call {@link CrossValidationUtils#setCurrentFold(int)}
	 * @param f
	 * @param folds
	 * @param nSamples
	 */
    public CrossValidationUtils(CatanDataSetIterator iter, int folds, int nSamples) {
    	if(folds < 2)
    		throw new RuntimeException("Cannot run crossValidation with less than 2 folds");
        this.iter = iter;
        this.nFolds = folds;
        this.nSamples = nSamples;
        fullData = new ArrayList<CatanDataSet>(nSamples);
        trainData = new ArrayList<CatanDataSet>(nSamples);
        init();
	}
	
    private void init(){
    	if(init)
    		return;
    	
    	dataSamples = 0;
    	while(iter.hasNext()){
    		CatanDataSet sample = iter.next(1);//0ne sample at a time
    		fullData.add(sample);
    		dataSamples++;
    		if(dataSamples == nSamples)
    			break;
    	}
    	
    	foldSize = dataSamples/nFolds;
    	foldStartIdx = 0; 
    	foldEndIdx = foldSize;
    	testData = new ArrayList<CatanDataSet>(foldSize);
    	
    	init = true;
    }
    
    public void setCurrentFold(int k){
    	if(k > nFolds && k < 0)
    		return;//TODO: report error?
    	foldEndIdx = foldSize*(k+1);
    	foldStartIdx = foldSize*k;
    	testData.clear();
    	trainData.clear();
         for(int idx = 0; idx < dataSamples; idx++){
			if(idx >= foldStartIdx && idx < foldEndIdx){
				//test data
				testData.add(fullData.get(idx));
			}else{
				//train data
				trainData.add(fullData.get(idx));
			}
        }
    }
	
    public ArrayList<CatanDataSet> getFullData() {
		return fullData;
	}
    
    public PreloadedCatanDataSetIterator getTrainIterator(){
    	return new PreloadedCatanDataSetIterator(trainData, iter.batch());
    }
    
    public PreloadedCatanDataSetIterator getTestIterator(){
    	return new PreloadedCatanDataSetIterator(testData, 1);
    }
    
    /**
	  * This is an ugly looking method and very repetitive due to the original design not anticipating the need for cross-validation
	  * TODO: find a more efficient approach that doesn't require so much code duplication
	  * @param plotter
	  * @param epochs
	  */
	 public void writeResults(CatanPlotter[] plotter, int task){
		 CatanPlotter resultsPlotter = new CatanPlotter(task);
		 
		 //evaluation accuracy
		 ArrayList<Double>[] list = new ArrayList[plotter.length];
		 for(int k = 0; k < plotter.length; k++){
			 list[k] = plotter[k].getEvalAccuracy();
		 }
		 double[][] ms = computeMeanAndStd(list);
		 ArrayList<Double> means = new ArrayList<Double>();
		 for(int i = 0; i < ms[0].length; i++){
			 means.add(ms[0][i]);
		 }
		 resultsPlotter.setEvalAccuracy(means);
		 resultsPlotter.writeEvalAcc(ms[1],ms[2]);
		 
		 //train accuracy
		 list = new ArrayList[plotter.length];
		 for(int k = 0; k < plotter.length; k++){
			 list[k] = plotter[k].getTrainAccuracy();
		 }
		 ms = computeMeanAndStd(list);
		 means = new ArrayList<Double>();
		 for(int i = 0; i < ms[0].length; i++){
			 means.add(ms[0][i]);
		 }
		 resultsPlotter.setTrainAccuracy(means);
		 resultsPlotter.writeTrainAcc(ms[1], ms[2]);
		 
		 //evaluation score
		 list = new ArrayList[plotter.length];
		 for(int k = 0; k < plotter.length; k++){
			 list[k] = plotter[k].getEvalScore();
		 }
		 ms = computeMeanAndStd(list);
		 means = new ArrayList<Double>();
		 for(int i = 0; i < ms[0].length; i++){
			 means.add(ms[0][i]);
		 }
		 resultsPlotter.setEvalScore(means);
		 resultsPlotter.writeEvalScore(ms[1], ms[2]);
		 
		 //train score
		 list = new ArrayList[plotter.length];
		 for(int k = 0; k < plotter.length; k++){
			 list[k] = plotter[k].getTrainScore();
		 }
		 ms = computeMeanAndStd(list);
		 means = new ArrayList<Double>();
		 for(int i = 0; i < ms[0].length; i++){
			 means.add(ms[0][i]);
		 }
		 resultsPlotter.setTrainScore(means);
		 resultsPlotter.writeTrainScore(ms[1], ms[2]);
		 
		 //evaluation/train ranks
		 ArrayList<INDArray>[] evalList = new ArrayList[plotter.length];
		 for(int k = 0; k < plotter.length; k++){
			 evalList[k] = plotter[k].getEvalRank();
		 }
		 ArrayList<INDArray>[] evalms = computeMeanNStd(evalList);
		 resultsPlotter.setEvalRank(evalms[0]);
		 
		 ArrayList<INDArray>[] trainList = new ArrayList[plotter.length];
		 for(int k = 0; k < plotter.length; k++){
			 trainList[k] = plotter[k].getTrainRank();
		 }
		 ArrayList<INDArray>[] trainms = computeMeanNStd(trainList);
		 resultsPlotter.setTrainRank(trainms[0]);
		 resultsPlotter.writeRanks(evalms[1], trainms[1]);
		 
		 //evaluation/train feature confusions
		 evalList = new ArrayList[plotter.length];
		 for(int k = 0; k < plotter.length; k++){
			 evalList[k] = plotter[k].getEvalFeatureConf();
		 }
		 evalms = computeMeanNStd(evalList);
		 resultsPlotter.setEvalFeatureConf(evalms[0]);
		 
		 trainList = new ArrayList[plotter.length];
		 for(int k = 0; k < plotter.length; k++){
			 trainList[k] = plotter[k].getTrainFeatureConf();
		 }
		 trainms = computeMeanNStd(trainList);
		 resultsPlotter.setTrainFeatureConf(trainms[0]);
		 resultsPlotter.writeFeatureConfusions(evalms[1], trainms[1]);
		 
	 }

	 /**
	  * three dimensional array, where the first dimension is the mean, the second is the standard deviation and the third the standard error
	  * @param list
	  * @return
	  */
	 private double[][] computeMeanAndStd(ArrayList<Double>[] list){
		 double[][] result = new double[3][list[0].size()];
		 for(int k = 0; k < list.length; k++){
			for(int i=0; i < list[k].size(); i++){
				result[0][i] += list[k].get(i);
			}
		 }
		 //divide to obtain the mean
		 for(int i=0; i < result[0].length; i++){
			 result[0][i] /= list.length;
		 }
		 //compute the standard deviation
		 for(int k = 0; k < list.length; k++){
			for(int i=0; i < list[k].size(); i++){
				result[1][i] += Math.pow(list[k].get(i) - result[0][i],2);
			}
		 }
		 //divide and sqrt to get the std deviation
		 for(int i=0; i < result[1].length; i++){
			 result[1][i] = Math.sqrt(result[1][i]/(list.length -1));
		 }
		 
		 //divide by the sqrt of the number of folds to get the standard error
		 for(int i=0; i < result[1].length; i++){
			 result[2][i] = result[1][i]/Math.sqrt(list.length -1);
		 }
		 
		 return result;
	 }
	 
	 /**
	  * three dimensional array, where the first dimension is the mean, the second is the standard deviation and the third the standard error
	  * @param list
	  * @return
	  */
	 private ArrayList<INDArray>[] computeMeanNStd(ArrayList<INDArray>[] list){
		 ArrayList<INDArray>[] result = new ArrayList[3];
		 result[0] = new ArrayList<INDArray>();
		 result[1] = new ArrayList<INDArray>();
		 result[2] = new ArrayList<INDArray>();
		 
		 //initialise all arrays
		 for(int i=0; i < list[0].size(); i++){
			 result[0].add(Nd4j.zeros(list[0].get(0).size(1)));
			 result[1].add(Nd4j.zeros(list[0].get(0).size(1)));
			 result[2].add(Nd4j.zeros(list[0].get(0).size(1)));
		 }
		 
		 //sum each
		 for(int k = 0; k < list.length; k++){
			for(int i=0; i < list[k].size(); i++){
				result[0].get(i).addiRowVector(list[k].get(i));
			}
		 }
		 //divide each to obtain the mean
		 for(int i=0; i < result[0].size(); i++){
			 result[0].get(i).divi(list.length);
		 }
		 //compute the standard deviation
		 for(int k = 0; k < list.length; k++){
			for(int i=0; i < list[k].size(); i++){
				result[1].get(i).addiRowVector(Transforms.pow(result[0].get(i).sub(list[k].get(i)),2));
			}
		 }
		 //divide and sqrt to get the std deviation
		 for(int i=0; i < result[1].size(); i++){
			 result[1].get(i).assign(Transforms.sqrt(result[1].get(i).divi(list.length -1)));
		 }
		 
		 //divide by the sqrt of the number of folds to get the standard error
		 for(int i=0; i < result[1].size(); i++){
			 result[2].get(i).assign(result[1].get(i).divi(Math.sqrt(list.length -1)));
		 }
		 
		 return result;
	 }
	 
}