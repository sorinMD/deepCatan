package test.seeding;

import java.io.File;
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
import util.ModelUtils;
import util.NNConfigParser;

/**
 * This the class used for training the best current model that will be used for seeding into MCTS (for the human data).
 * It uses all the data for the task and the configuration is the one suggested by the previous results.
 * 
 * @author MD
 *
 */
public class BestMlpTest {
	private static Logger log = LoggerFactory.getLogger(BestMlpTest.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static NNConfigParser parser;
	private static int nTasks = 6;
	private static int TASK = 0;
	private static boolean NORMALISATION;
	
	public static void main(String[] args) throws Exception {
    	parser = new NNConfigParser(null);//TODO: add config file to resources and accept different ones via the args
    	DATA_TYPE = parser.getDataType();
    	DATA_TYPE = DATA_TYPE + "CV"; //the unsplit files are all in this folder
    	PATH = parser.getDataPath();
    	TASK = parser.getTask();
    	NORMALISATION = parser.getNormalisation();
    	
    	//the features used are the same across the tasks so we just use the final ones, but we need to read them as they are present in all metadata files
    	int numInputs = 0;
        int actInputSize = 0;
        int[] maxActions = new int[nTasks];//this is different for each task;
    	
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
	        maxActions[i] = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
	        scanner.close();
    	}
    	
        //Set specific params
    	int nSamples = parser.getNumberOfSamples();
        int epochs = parser.getEpochs();
        int miniBatchSize = parser.getMiniBatchSize(); 
        double labelW = parser.getLabelWeight();
        double metricW = parser.getMetricWeight();
        double learningRate = parser.getLearningRate();
        
        //debugging params (these can be set here)
        int iterations = 1; //iterations over each batch
        long seed = 123;
       
        //data iterators
        CatanDataSetIterator fullIter = new CatanDataSetIterator(data[TASK],nSamples,miniBatchSize,numInputs+1,numInputs+actInputSize+1,true,parser.getMaskHiddenFeatures());
        
        Normaliser norm = new Normaliser(PATH + DATA_TYPE,parser.getMaskHiddenFeatures());
        if(NORMALISATION){
        	log.info("Check normalisation parameters for dataset ....");
        	norm.init(fullIter, TASK);
        }
        
        //if the input is masked/postprocessed update the input size for the model creation
        if(parser.getMaskHiddenFeatures()) {
        	numInputs -= CatanFeatureMaskingUtil.droppedFeaturesCount;
        	actInputSize -= CatanFeatureMaskingUtil.droppedFeaturesCount;
        }
        
        log.info("Pre-loading data to avoid the MemcpyAsync bug....");
        CrossValidationUtils temp = new CrossValidationUtils(fullIter, 10, nSamples);//only to get access to the full data
        PreloadedCatanDataSetIterator preloadedIter = new PreloadedCatanDataSetIterator(temp.getFullData(), miniBatchSize);        
        
        log.info("Build model....");
        //remove the biases, since dl4j adds it's own and the iterator removes the bias from the data
        CatanMlpConfig modelConfig = new CatanMlpConfig(numInputs + actInputSize-2, 1, seed, iterations, WeightInit.XAVIER, Updater.RMSPROP, learningRate, LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        CatanMlp network = modelConfig.init();
    	
        CatanEvaluation tEval;
        ActionGuessFeaturesEvaluation tagfe;
        CatanPlotter plotter = new CatanPlotter(TASK);
        INDArray output;
        for( int i=0; i<epochs; i++ ) {
        	while(preloadedIter.hasNext()){
        		CatanDataSet d = preloadedIter.next();
        		DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
        		network.fit(ds,d.getSizeOfLegalActionSet());//set the labels we want to fit to
        	}
        	preloadedIter.reset();
            log.info("*** Completed epoch {} ***", i);
            
            log.info("Saving network parameters");
            ModelUtils.saveMlpModelAndParameters(network, TASK);
            
            log.info("Evaluate model on training set....");
            tEval = new CatanEvaluation(maxActions[TASK]);
            tagfe = new ActionGuessFeaturesEvaluation(actInputSize-1);//get rid of the bias
            while (preloadedIter.hasNext()) {
            	CatanDataSet d = preloadedIter.next(1);
            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
            	output = network.output(ds, d.getSizeOfLegalActionSet());
	            tEval.eval(DataUtils.computeLabels(d), output.transposei());
	            tagfe.eval(d.getTargetAction(), d.getActionFeatures().getRow(Nd4j.getBlasWrapper().iamax(output)));
            }
            preloadedIter.reset();
            //once finished just output the result
            System.out.println(tEval.stats());
            System.out.println("Feature confusion: " + Arrays.toString(tagfe.getStats()));
            plotter.addData(0, tEval.score(), 0, tEval.accuracy());
            plotter.addRanks(tEval.getRank(), tEval.getRank());
            plotter.addFeatureConfusion(tagfe.getStats(), tagfe.getStats());
            plotter.writeAll();
        }
        
    }  

}

