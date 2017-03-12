package test.catan.mlp;

import java.io.File;
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
	        fullIter[i] = new CatanDataSetIterator(data[i],nSamples,miniBatchSize,numInputs+1,numInputs+actInputSize+1,true);
        }
        
        Normaliser norm = new Normaliser(PATH + DATA_TYPE);
        if(NORMALISATION){
        	log.info("Check normalisation parameters for dataset ....");
        	norm.init(fullIter);
        }
        
        log.info("Initialise cross-validation for each task....");
        CrossValidationUtils[] cv = new CrossValidationUtils[nTasks];
        for(int i= 0; i < nTasks; i++){
        	cv[i] = new CrossValidationUtils(fullIter[i], FOLDS, nSamples);
        }
        //NOTE: when training we iterate for each fold over each task, but we need to be able to pass the data for each task over each fold, hence this order
        CatanPlotter[][] plotter = new CatanPlotter[nTasks][FOLDS];
        
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
	            		CatanDataSet d = trainIter[j].next();
	            		DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
	                	if(NORMALISATION)
	                		norm.normalizeZeroMeanUnitVariance(ds);
	            		network.fit(ds,d.getSizeOfLegalActionSet());//set the labels we want to fit to
	            	}
	            }
//	            log.info("*** Completed epoch {} ***", i);
	            
//	            log.info("Evaluate model....");
	            for(int j= 0; j < nTasks; j++){
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
	        }
            //write current results and move files so these won't be overwritten
            for(int j= 0; j < nTasks; j++){
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
    		cv[i].writeResults(plotter[i], i);
    	}
        
    }  
}

