package test.catan.mlp;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.ActionGuessFeaturesEvaluation;
import org.deeplearning4j.eval.CatanEvaluation;
import org.deeplearning4j.eval.CatanPlotter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.multilayer.CatanMlp;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import data.CatanDataSet;
import data.CatanDataSetIterator;
import data.Normaliser;
import data.PreloadedCatanDataSetIterator;
import model.CatanMlpConfig;
import util.CatanFeatureMaskingUtil;
import util.CrossValidationUtils;
import util.DataUtils;
import util.NNConfigParser;


public class CVMlpFullDataSetTest {
	private static Logger log = LoggerFactory.getLogger(CVMlpFullDataSetTest.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static NNConfigParser parser;
	private static int nTasks = 6;
	private static boolean NORMALISATION;
	
    public static void main(String[] args) throws Exception {
    	parser = new NNConfigParser(null);
    	DATA_TYPE = parser.getDataType();
    	DATA_TYPE = DATA_TYPE + "CV";
    	PATH = parser.getDataPath();
    	NORMALISATION = parser.getNormalisation();
    	int FOLDS = 10; //standard 10 fold validation
    	
    	//the features used are the same across the tasks so we just use the final ones, but we need to read them as they are present in all metadata files
    	int numInputs = 0;
        int actInputSize = 0;
        int[] trainMaxActions = new int[nTasks];//this is different for each task;
    	
    	log.info("Load data for each task....");
    	File[] data = new File[nTasks];
    	for(int i= 0; i < nTasks; i++){
    		data[i] = new File(PATH + DATA_TYPE + "/alldata-" + i + ".txt");
	        //read metadata and feed in all the required info
	    	Scanner scanner = new Scanner(new File(PATH + DATA_TYPE + "/alldata-" + i + "-metadata.txt"));
	    	if(!scanner.hasNextLine()){
	    		scanner.close();
	    		throw new RuntimeException("Metadata not found; Cannot initialise network parameters");
	    	}
		    	numInputs = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
		        actInputSize = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
		        trainMaxActions[i] = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
		        scanner.close();
    	}
    	
        int nSamples = parser.getNumberOfSamples();
        
        //Set specific params
        int epochs = parser.getEpochs();
        int miniBatchSize = parser.getMiniBatchSize(); 
        double labelW = parser.getLabelWeight();
        double metricW = parser.getMetricWeight();
        double learningRate = parser.getLearningRate();
        
        //debugging params (these can be set here)
        int iterations = 1; //iterations over each batch
        long seed = 123;
       
        //data iterators
        CatanDataSetIterator[] fullIter = new CatanDataSetIterator[nTasks];
        for(int i= 0; i < nTasks; i++){
	        fullIter[i] = new CatanDataSetIterator(data[i],nSamples,miniBatchSize,numInputs+1,numInputs+actInputSize+1,true,parser.getMaskHiddenFeatures());
        }
        
        Normaliser norm = new Normaliser(PATH + DATA_TYPE,parser.getMaskHiddenFeatures());
        if(NORMALISATION){
        	log.info("Check normalisation parameters for dataset ....");
        	norm.init(fullIter);
        }
        
        //if the input is masked/postprocessed update the input size for the model creation
        if(parser.getMaskHiddenFeatures()) {
        	numInputs -= CatanFeatureMaskingUtil.droppedFeaturesCount;
        	actInputSize -= CatanFeatureMaskingUtil.droppedFeaturesCount;
        }
        
        log.info("Initialise cross-validation for each task....");
        CrossValidationUtils[] cv = new CrossValidationUtils[nTasks];
        for(int i= 0; i < nTasks; i++){
        	cv[i] = new CrossValidationUtils(fullIter[i], FOLDS, nSamples);
        }
        //NOTE: when training we iterate for each fold over each task, but we need to be able to pass the data for each task over each fold, hence this order
        CatanPlotter[][] plotter = new CatanPlotter[nTasks][FOLDS];
        double[] totalEvalAccuracy = new double[epochs];
        Arrays.fill(totalEvalAccuracy, 0);
                
        for(int k = 0; k < FOLDS; k++){
        	log.info("Starting fold " + k);
        	for(int i= 0; i < nTasks; i++){
        		cv[i].setCurrentFold(k);
        	}
//	        log.info("Load test data for each task....");
	    	PreloadedCatanDataSetIterator[] trainIter = new PreloadedCatanDataSetIterator[nTasks];
	    	PreloadedCatanDataSetIterator[] testIter = new PreloadedCatanDataSetIterator[nTasks];
	        CatanEvaluation[] eval = new CatanEvaluation[nTasks];
	        CatanEvaluation[] tEval = new CatanEvaluation[nTasks];
	        ActionGuessFeaturesEvaluation[] tagfe = new ActionGuessFeaturesEvaluation[nTasks];
	        ActionGuessFeaturesEvaluation[] agfe = new ActionGuessFeaturesEvaluation[nTasks];;
	        
	        for(int i= 0; i < nTasks; i++){
		        trainIter[i] = cv[i].getTrainIterator();
	        	testIter[i] = cv[i].getTestIterator();
		        plotter[i][k] = new CatanPlotter(i);
		        eval[i] = new CatanEvaluation(trainMaxActions[i]);
		        tEval[i] = new CatanEvaluation(trainMaxActions[i]);
	        }
	        
//	        log.info("Build model....");
	        //remove the biases, since dl4j adds it's own and the iterator removes the bias from the data
	        CatanMlpConfig netConfig = new CatanMlpConfig(numInputs + actInputSize-2, 1, seed, iterations, WeightInit.XAVIER, Updater.RMSPROP, learningRate, LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
	        CatanMlp network = netConfig.init();
	        
	        INDArray output;
//	        log.info("Train model....");
	        for( int i=0; i<epochs; i++ ) {
	        	//fitting one batch from each task repeatedly until we cover all samples
		        while (trainIter[2].hasNext()) { //task 2 always has the most data
		        	for(int j= 0; j < nTasks; j++){
		        		if(!trainIter[j].hasNext())
		        			continue;
		        		if(j == 4 && parser.getMaskHiddenFeatures())//we can't train on discard task when the input is masked since there is no difference between the actions
		        			continue;
	            		CatanDataSet d = trainIter[j].next();
	            		DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
	                	if(NORMALISATION)
	                		norm.normalizeZeroMeanUnitVariance(ds);
	            		network.fit(ds,d.getSizeOfLegalActionSet());//set the labels we want to fit to
	            	}
	            }
//	            log.info("*** Completed epoch {} ***", i);
	            
//	            log.info("Evaluate model....");
	            double totalCorrect = 0;
	            double totalSamples = 0;
	            for(int j= 0; j < nTasks; j++){
	        		if(j == 4 && parser.getMaskHiddenFeatures())//we can't evaluate on discard task when the input is masked since there is no difference between the actions
	        			continue;
//	            	log.info("Evaluate model on evaluation set for task " + j + "....");
		            eval[j] = new CatanEvaluation(trainMaxActions[j]);
		            agfe[j] = new ActionGuessFeaturesEvaluation(actInputSize-1);//get rid of the bias
		            while (testIter[j].hasNext()) {
		            	CatanDataSet d = testIter[j].next();
		            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
		            	if(NORMALISATION)
		            		norm.normalizeZeroMeanUnitVariance(ds);
		            	output = network.output(ds, d.getSizeOfLegalActionSet());
			            eval[j].eval(DataUtils.computeLabels(d), output.transposei());
			            agfe[j].eval(d.getTargetAction(), d.getActionFeatures().getRow(Nd4j.getBlasWrapper().iamax(output)));
		            }
		            testIter[j].reset();
		            //once finished just output the result
//		            System.out.println(eval[j].stats());
		            //do the below across all tasks to average as in the standard approach
		            totalCorrect += eval[j].correct();
		            totalSamples += eval[j].getNumRowCounter();
		            		            
//		            log.info("Evaluate model on training set for task " + j + "....");
		            tEval[j] = new CatanEvaluation(trainMaxActions[j]);
		            tagfe[j] = new ActionGuessFeaturesEvaluation(actInputSize-1);//get rid of the bias
		            trainIter[j].reset();
		            while (trainIter[j].hasNext()) {
		            	CatanDataSet d = trainIter[j].next(1);
		            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
		            	if(NORMALISATION)
		            		norm.normalizeZeroMeanUnitVariance(ds);
		            	output = network.output(ds, d.getSizeOfLegalActionSet());
			            tEval[j].eval(DataUtils.computeLabels(d), output.transposei());
			            tagfe[j].eval(d.getTargetAction(), d.getActionFeatures().getRow(Nd4j.getBlasWrapper().iamax(output)));
		            }
		            trainIter[j].reset();
		            //once finished just output the result
//		            System.out.println(tEval[j].stats());
		            plotter[j][k].addData(eval[j].score(), tEval[j].score(), eval[j].accuracy(), tEval[j].accuracy());
		            plotter[j][k].addRanks(tEval[j].getRank(), eval[j].getRank());
		            plotter[j][k].addFeatureConfusion(agfe[j].getStats(), tagfe[j].getStats());
	            }
	            totalEvalAccuracy[i] += totalCorrect/totalSamples;
	        }
            //write current results and move files so these won't be overwritten
            for(int j= 0; j < nTasks; j++){
            	if(j == 4 && parser.getMaskHiddenFeatures())//there is nothing to write so we should avoid an exception
        			continue;
	            plotter[j][k].writeAll();
            }
            File dirFrom = new File(CatanPlotter.RESULTS_DIR);
            File[] files = dirFrom.listFiles();
            File dirTo = new File("" + k);
            dirTo.mkdirs();
            for(File f : files){
            	FileUtils.moveFileToDirectory(f, dirTo, false);
            }
	        
        }
        
    	for(int i= 0; i < nTasks; i++){
    		if(i == 4 && parser.getMaskHiddenFeatures())
    			continue;
    		cv[i].writeResults(plotter[i], i);
    	}
        
    	//write the results for the total evaluation accuracy
    	for(int i= 0; i < totalEvalAccuracy.length; i++){
	        totalEvalAccuracy[i] /= FOLDS;
    	}
        
		try {
            File write = new File("TotalEvaluation.txt");
			write.delete();
			write.createNewFile();
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
            StringBuilder sb = new StringBuilder();
            for(Object value : totalEvalAccuracy) {
                sb.append(String.format("%.10f", (Double) value));
                sb.append(",");
            }
            String line = sb.toString();
            line = line.substring(0, line.length()-1);
            bos.write(line.getBytes());
            bos.write(line.getBytes());
            bos.flush();
            bos.close();

        } catch(IOException e){
            throw new RuntimeException(e);
        }
     
    }  
}

