package test.seeding;

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
import data.PreloadedCatanDataSetIterator;
import model.CatanMlpConfig;
import util.CatanFeatureMaskingUtil;
import util.CrossValidationUtils;
import util.DataUtils;
import util.ModelUtils;
import util.NNConfigParser;

/**
 * Best model trained on all the tasks at once that would be used for seeding the MCTS agent (for the human data).
 * It uses the alldata files.
 * 
 * @author sorinMD
 *
 */
public class BestSingleMlpTest {

	private static Logger log = LoggerFactory.getLogger(BestSingleMlpTest.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static NNConfigParser parser;
	private static int nTasks = 6;
	private static boolean NORMALISATION;
	
    public static void main(String[] args) throws Exception {
    	parser = new NNConfigParser(null);//TODO: add config file to resources and accept different ones via the args
    	DATA_TYPE = parser.getDataType() + "CV"; //the unsplit human files are all in this folder
    	PATH = parser.getDataPath();
    	NORMALISATION = parser.getNormalisation();
    	
    	//the features used are the same across the tasks so we just use the final ones, but we need to read them as they are present in all metadata files
    	int numInputs = 0;
        int actInputSize = 0;
        int[] trainMaxActions = new int[nTasks];//this is different for each task;
    	
    	log.info("Load train data for each task....");
    	File[] trainData = new File[nTasks];
    	for(int i= 0; i < nTasks; i++){
    		trainData[i] = new File(PATH + DATA_TYPE + "/alldata-" + i + ".txt");
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
        CatanDataSetIterator[] trainIter = new CatanDataSetIterator[nTasks];
        
        for(int i= 0; i < nTasks; i++){
	        trainIter[i] = new CatanDataSetIterator(trainData[i],nSamples,miniBatchSize,numInputs+1,numInputs+actInputSize+1,true,parser.getMaskHiddenFeatures());
        }
        
        Normaliser norm = new Normaliser(PATH + DATA_TYPE,parser.getMaskHiddenFeatures());
        if(NORMALISATION){
        	log.info("Check normalisation parameters for dataset ....");
        	norm.init(trainIter);
        }
        
        //if the input is masked/postprocessed update the input size for the model creation
        if(parser.getMaskHiddenFeatures()) {
        	numInputs -= CatanFeatureMaskingUtil.droppedFeaturesCount;
        	actInputSize -= CatanFeatureMaskingUtil.droppedFeaturesCount;
        }
        
        //and remove the bias count, since dl4j adds it's own and the iterator removes the bias from the data
        numInputs--;
        actInputSize--;
        
        log.info("Pre-loading data to avoid the MemcpyAsync bug on GPU....");
        CrossValidationUtils[] temp = new CrossValidationUtils[nTasks];
        PreloadedCatanDataSetIterator[] preloadedIter = new PreloadedCatanDataSetIterator[nTasks];
        for(int j = 0; j < nTasks; j++) {
        	temp[j] = new CrossValidationUtils(trainIter[j], 10, nSamples);//only to get access to the full data
        	preloadedIter[j] = new PreloadedCatanDataSetIterator(temp[j].getFullData(), miniBatchSize);
        }
        
        log.info("Build model....");
        CatanMlpConfig netConfig = new CatanMlpConfig(numInputs + actInputSize, 1, seed, iterations, WeightInit.XAVIER, Updater.RMSPROP, learningRate, LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        CatanMlp network = netConfig.init();
        
//      //for debugging
//        int listenerFreq = 1;
//        LossPlotterIterationListener lossListener = new LossPlotterIterationListener(listenerFreq);
//        network.setListeners(new ScoreIterationListener(listenerFreq), lossListener);
        
        CatanPlotter[] plotter = new CatanPlotter[nTasks];
        CatanEvaluation[] eval = new CatanEvaluation[nTasks];
        CatanEvaluation[] tEval = new CatanEvaluation[nTasks];
    	
        for(int i= 0; i < nTasks; i++){
	        plotter[i] = new CatanPlotter(i);
	        eval[i] = new CatanEvaluation(trainMaxActions[i]);//placeholder
        }
        
        INDArray output;
        
        log.info("Train model....");
        for( int i=0; i<epochs; i++ ) {
        	//fitting one batch from each task repeatedly until we cover all samples
        	while (hasNext(preloadedIter)) { 
	        	for(int j= 0; j < nTasks; j++){
	        		if(!preloadedIter[j].hasNext())
	        			continue;
	        		if(j == 4 && parser.getMaskHiddenFeatures())//we can't train on discard task when the input is masked since there is no difference between the actions
	        			continue;
            		CatanDataSet d = preloadedIter[j].next();
            		DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
                	if(NORMALISATION)
                		norm.normalizeZeroMeanUnitVariance(ds);
            		network.fit(ds,d.getSizeOfLegalActionSet());//set the labels we want to fit to
            	}
            }
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            for(int j= 0; j < nTasks; j++){
	            log.info("Evaluate model on training set for task " + j + "....");
        		if(j == 4 && parser.getMaskHiddenFeatures())//we can't evaluate on discard task when the input is masked since there is no difference between the actions
        			continue;
	            tEval[j] = new CatanEvaluation(trainMaxActions[j]);
	            preloadedIter[j].reset();
	            while (preloadedIter[j].hasNext()) {
	            	CatanDataSet d = preloadedIter[j].next(1);
	            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
	            	if(NORMALISATION)
	            		norm.normalizeZeroMeanUnitVariance(ds);
	            	output = network.output(ds, d.getSizeOfLegalActionSet());
		            tEval[j].eval(DataUtils.computeLabels(d), output.transposei());
	            }
	            preloadedIter[j].reset();
	            //once finished just output the result
	            System.out.println(tEval[j].stats());
	            plotter[j].addData(eval[j].score(), tEval[j].score(), eval[j].accuracy(), tEval[j].accuracy());
	            plotter[j].addRanks(tEval[j].getRank(), eval[j].getRank());
//            	plotter.plotAll();
	            plotter[j].writeAll();
            }
            
        	log.info("Saving network parameters");
        	ModelUtils.saveMlpModelAndParameters(network, 6);//all tasks together are represented as task 6

        }
    }
    
	private static boolean hasNext(PreloadedCatanDataSetIterator[] fullIter){
        for(int j= 0; j < nTasks; j++){
    		if(j == 4 && parser.getMaskHiddenFeatures())
    			continue;
        	if(fullIter[j].hasNext())
        		return true;
        }
		return false;	
	}
}
