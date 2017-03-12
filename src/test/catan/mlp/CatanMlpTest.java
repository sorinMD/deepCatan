package test.catan.mlp;

import java.io.File;
import java.util.Scanner;

import model.CatanMlpConfig;
import util.DataUtils;
import util.ModelUtils;
import util.NNConfigParser;

import org.deeplearning4j.eval.CatanEvaluation;
import org.deeplearning4j.eval.CatanPlotter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.multilayer.CatanMlp;
import org.deeplearning4j.nn.weights.WeightInit;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import data.CatanDataSet;
import data.CatanDataSetIterator;
import data.Normaliser;

/**
 * Training and testing the Catan Mlp model without cross-validation.
 * 
 * @author sorinMD
 *
 */
public class CatanMlpTest {

	private static Logger log = LoggerFactory.getLogger(CatanMlpTest.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static int TASK;
	private static NNConfigParser parser;
	private static boolean NORMALISATION;
	
    public static void main(String[] args) throws Exception {
    	
    	parser = new NNConfigParser(null);//TODO: add config file to resources and accept different ones via the args
    	TASK = parser.getTask();
    	DATA_TYPE = parser.getDataType();
    	PATH = parser.getDataPath();
    	NORMALISATION = parser.getNormalisation();
    	
    	log.info("Load train data....");
    	//the file containing the training data
    	File trainData = new File(PATH + DATA_TYPE + "/train-" + TASK + ".txt");
        //read metadata and feed in all the required info
    	Scanner scanner = new Scanner(new File(PATH + DATA_TYPE + "/train-" + TASK + "-metadata.txt"));
    	if(!scanner.hasNextLine()){
    		scanner.close();
    		throw new RuntimeException("Metadata not found; Cannot initialise network parameters");
    	}
    	int numInputs = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
        int actInputSize = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
        int trainMaxActions = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
        int nSamples = parser.getNumberOfSamples();
        scanner.close();
        
        //Set specific params
        int epochs = parser.getEpochs();
        int miniBatchSize = parser.getMiniBatchSize(); 
        double labelW = parser.getLabelWeight();
        double metricW = parser.getMetricWeight();
        double learningRate = parser.getLearningRate();
        
        //debugging params (these can be set here)
        int iterations = 1; //iterations over each batch
        long seed = 123;
       
        //data iterator
        CatanDataSetIterator trainIter = new CatanDataSetIterator(trainData,nSamples,miniBatchSize,numInputs+1,numInputs+actInputSize+1,true);
        Normaliser norm = new Normaliser(PATH + DATA_TYPE);
        if(NORMALISATION){
        	log.info("Check normalisation parameters for dataset ....");
        	norm.init(trainIter, TASK);
        }
        
        //and remove the bias count, since dl4j adds it's own and the iterator removes the bias from the data
        numInputs--;
        actInputSize--;
        
        log.info("Build model....");
        CatanMlpConfig netConfig = new CatanMlpConfig(numInputs + actInputSize, 1, seed, iterations, WeightInit.XAVIER, Updater.RMSPROP, learningRate, LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        CatanMlp network = netConfig.init();

//      //debugging
//        LossPlotterIterationListener lossListener = new LossPlotterIterationListener(listenerFreq);
//        int listenerFreq = 1;
//        network.setListeners(new ScoreIterationListener(listenerFreq));
        
        log.info("Load test data....");
        //get the test data from completely different games
        File testData = new File(PATH + DATA_TYPE + "/test-" + TASK + ".txt");
    	Scanner testScanner = new Scanner(new File(PATH + DATA_TYPE + "/test-" + TASK + "-metadata.txt"));
    	if(!testScanner.hasNextLine()){
    		testScanner.close();
    		throw new RuntimeException("Metadata not found; Cannot initialise evaluation set");
    	}
    	
    	numInputs = Integer.parseInt(testScanner.nextLine().split(METADATA_SEPARATOR)[1]);
        actInputSize = Integer.parseInt(testScanner.nextLine().split(METADATA_SEPARATOR)[1]);
        int maxActions = Integer.parseInt(testScanner.nextLine().split(METADATA_SEPARATOR)[1]);
        testScanner.close();
        //use 10k unseen samples to evaluate (equal to all samples for human data) 
        CatanDataSetIterator testIter = new CatanDataSetIterator(testData,10000,1,numInputs+1,numInputs+actInputSize+1,true);
        
        CatanPlotter plotter = new CatanPlotter(parser.getTask());
        CatanEvaluation eval = new CatanEvaluation(maxActions);
        CatanEvaluation tEval = new CatanEvaluation(trainMaxActions);
        INDArray output;
               
        log.info("Train model....");
        INDArray branch = Nd4j.zeros(1);
        for( int i=0; i<epochs; i++ ) {
        	trainIter.reset();
        	int nS = 0;
        	while(trainIter.hasNext()){
        		nS++;
        		CatanDataSet d = trainIter.next();
        		branch.addi(d.getSizeOfLegalActionSet());
        		DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
        		network.fit(ds,d.getSizeOfLegalActionSet());//set the labels we want to fit to
        	}
            log.info("*** Completed epoch {} ***", i);
            System.out.println("Avg branch is " + branch.div(nS));
            
            log.info("Saving network parameters", i);
        	ModelUtils.saveMlpModelAndParameters(network, TASK);
            
            log.info("Evaluate model on evaluation set....");
            eval = new CatanEvaluation(maxActions);
            while (testIter.hasNext()) {
            	CatanDataSet d = testIter.next();
            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
            	output = network.output(ds, d.getSizeOfLegalActionSet());
	            eval.eval(DataUtils.computeLabels(d), output.transposei());
            }
            testIter.reset();
            //once finished just output the result
            System.out.println(eval.stats());
            
            log.info("Evaluate model on training set....");
            tEval = new CatanEvaluation(trainMaxActions);
            trainIter.reset();
            while (trainIter.hasNext()) {
            	CatanDataSet d = trainIter.next(1);
            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
            	output = network.output(ds, d.getSizeOfLegalActionSet());
	            tEval.eval(DataUtils.computeLabels(d), output.transposei());
            }
            trainIter.reset();
            //once finished just output the result
            System.out.println(tEval.stats());
            
            plotter.addData(eval.score(), tEval.score(), eval.accuracy(), tEval.accuracy());
            plotter.addRanks(tEval.getRank(), eval.getRank());
//            plotter.plotAll();
            plotter.writeAll();
        
        }
        
    }      

}
