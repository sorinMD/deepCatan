package test.catan.mlp;

import java.io.File;
import java.util.Scanner;
import org.deeplearning4j.eval.CatanEvaluation;
import org.deeplearning4j.eval.CatanPlotter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.multilayer.CatanMlp;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import data.CatanDataSet;
import data.CatanDataSetIterator;
import data.Normaliser;
import model.CatanMlpConfig;
import util.DataUtils;
import util.ModelUtils;
import util.NNConfigParser;

public class MlpFullDataSetTest {

	private static Logger log = LoggerFactory.getLogger(MlpFullDataSetTest.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static NNConfigParser parser;
	private static int nTasks = 6;
	private static boolean NORMALISATION;
	
    public static void main(String[] args) throws Exception {
    	parser = new NNConfigParser(null);
    	DATA_TYPE = parser.getDataType();
    	PATH = parser.getDataPath();
    	NORMALISATION = parser.getNormalisation();
    	
    	//the features used are the same across the tasks so we just use the final ones, but we need to read them as they are present in all metadata files
    	int numInputs = 0;
        int actInputSize = 0;
        int[] trainMaxActions = new int[nTasks];//this is different for each task;
    	
    	log.info("Load train data for each task....");
    	File[] trainData = new File[nTasks];
    	for(int i= 0; i < nTasks; i++){
    		trainData[i] = new File(PATH + DATA_TYPE + "/train-" + i + ".txt");
	        //read metadata and feed in all the required info
	    	Scanner scanner = new Scanner(new File(PATH + DATA_TYPE + "/train-" + i + "-metadata.txt"));
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
        CatanDataSetIterator[] trainIter = new CatanDataSetIterator[nTasks];
        
        for(int i= 0; i < nTasks; i++){
	        trainIter[i] = new CatanDataSetIterator(trainData[i],nSamples,miniBatchSize,numInputs+1,numInputs+actInputSize+1,true);
        }
        
        Normaliser norm = new Normaliser(PATH + DATA_TYPE);
        if(NORMALISATION){
        	log.info("Check normalisation parameters for dataset ....");
        	norm.init(trainIter);
        }
        
        //and remove the bias count, since dl4j adds it's own and the iterator removes the bias from the data
        numInputs--;
        actInputSize--;
        
        log.info("Build model....");
        CatanMlpConfig netConfig = new CatanMlpConfig(numInputs + actInputSize, 1, seed, iterations, WeightInit.XAVIER, Updater.RMSPROP, learningRate, LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        CatanMlp network = netConfig.init();     
        
//      //for debugging
//        int listenerFreq = 1;
//        LossPlotterIterationListener lossListener = new LossPlotterIterationListener(listenerFreq);
//        network.setListeners(new ScoreIterationListener(listenerFreq), lossListener);
        
        log.info("Load test data for each task....");
        //get the test data from completely different games
        File[] testData = new File[nTasks];
        int maxActions[] = new int[nTasks];//this is different for each task;
        for(int i= 0; i < nTasks; i++){
	        testData[i] = new File(PATH + DATA_TYPE + "/test-" + i + ".txt");
	    	Scanner scanner = new Scanner(new File(PATH + DATA_TYPE + "/test-" + i + "-metadata.txt"));
	    	if(!scanner.hasNextLine()){
	    		scanner.close();
	    		throw new RuntimeException("Metadata not found; Cannot initialise evaluation set");
	    	}
	    	
	    	numInputs = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
	        actInputSize = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
	    	maxActions[i] = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
	    	scanner.close();
        }
        
        CatanDataSetIterator testIter[] = new CatanDataSetIterator[nTasks];
        CatanPlotter[] plotter = new CatanPlotter[nTasks];
        CatanEvaluation[] eval = new CatanEvaluation[nTasks];
        CatanEvaluation[] tEval = new CatanEvaluation[nTasks];
    	
        for(int i= 0; i < nTasks; i++){
        	//use 10k of unseen data to evaluate (or the equivalent to all the human data)
	        testIter[i] = new CatanDataSetIterator(testData[i],10000,1,numInputs+1,numInputs+actInputSize+1,true);
	        plotter[i] = new CatanPlotter(i);
	        eval[i] = new CatanEvaluation(trainMaxActions[i]);
	        tEval[i] = new CatanEvaluation(maxActions[i]);
        }
        
        INDArray output;
        
        log.info("Train model....");
        for( int i=0; i<epochs; i++ ) {
        	//fitting one batch from each task repeatedly until we cover all samples
        	//TODO: make sure to comment out the corresponding one
//        	while (trainIter[2].hasNext()) { //human data where task 2 has the most data
        	while (trainIter[0].hasNext() && trainIter[1].hasNext() && trainIter[2].hasNext() && trainIter[3].hasNext() && trainIter[4].hasNext() && trainIter[5].hasNext()) { //synth data where we need to keep them the same size
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
            log.info("*** Completed epoch {} ***", i);

            log.info("Saving network parameters", i);
        	ModelUtils.saveMlpModelAndParameters(network, 6);//all tasks together are represented as task 6
            
            log.info("Evaluate model....");
            for(int j= 0; j < nTasks; j++){
            	log.info("Evaluate model on evaluation set for task " + j + "....");
	            eval[j] = new CatanEvaluation(maxActions[j]);
	            while (testIter[j].hasNext()) {
	            	CatanDataSet d = testIter[j].next();
	            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
	            	if(NORMALISATION)
	            		norm.normalizeZeroMeanUnitVariance(ds);
	            	output = network.output(ds, d.getSizeOfLegalActionSet());
		            eval[j].eval(DataUtils.computeLabels(d), output.transposei());
	            }
	            testIter[j].reset();
	            //once finished just output the result
	            System.out.println(eval[j].stats());
	            
	            log.info("Evaluate model on training set for task " + j + "....");
	            tEval[j] = new CatanEvaluation(trainMaxActions[j]);
	            trainIter[j].reset();
	            while (trainIter[j].hasNext()) {
	            	CatanDataSet d = trainIter[j].next(1);
	            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
	            	if(NORMALISATION)
	            		norm.normalizeZeroMeanUnitVariance(ds);
	            	output = network.output(ds, d.getSizeOfLegalActionSet());
		            tEval[j].eval(DataUtils.computeLabels(d), output.transposei());
	            }
	            trainIter[j].reset();
	            //once finished just output the result
	            System.out.println(tEval[j].stats());
	            plotter[j].addData(eval[j].score(), tEval[j].score(), eval[j].accuracy(), tEval[j].accuracy());
	            plotter[j].addRanks(tEval[j].getRank(), eval[j].getRank());
//            	plotter.plotAll();
	            plotter[j].writeAll();
            }
        }
    }  
}
