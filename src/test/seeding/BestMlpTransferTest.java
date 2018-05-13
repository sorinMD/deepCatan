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
 * This the class used for training the best current model (trained on human data) that will be used for seeding into MCTS.
 * It uses all the data for the task and the configuration is the one suggested by the previous results.
 * 
 * @author MD
 *
 */
public class BestMlpTransferTest {
	private static Logger log = LoggerFactory.getLogger(BestMlpTransferTest.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static NNConfigParser parser;
	private static int nTasks = 6;
	private static int TASK = 0;
	private static boolean NORMALISATION;
	/**
	 * The number of epochs the system is trained on all the data from the tasks we are trying to transfer from
	 */
	private static int preEpochs = 1;
	
	public static void main(String[] args) throws Exception {
    	parser = new NNConfigParser(null);//TODO: add config file to resources and accept different ones via the args
    	DATA_TYPE = parser.getDataType();
    	DATA_TYPE = DATA_TYPE + "CV";
    	PATH = parser.getDataPath();
    	TASK = parser.getTask();
    	NORMALISATION = parser.getNormalisation();
    	preEpochs = parser.getPreTrainEpochs();
    	
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
        
        //TODO: Should we normalise over the whole data or just on this particular task that we want to handle or just the 5 tasks?
        Normaliser norm = new Normaliser(PATH + DATA_TYPE,parser.getMaskHiddenFeatures());
        if(NORMALISATION){
        	log.info("Check normalisation parameters for dataset ....");
        	norm.init(fullIter);
        }
        
        //if the input is masked/postprocessed update the input size for the models creation
        if(parser.getMaskHiddenFeatures()) {
        	numInputs -= CatanFeatureMaskingUtil.droppedFeaturesCount;
        	actInputSize -= CatanFeatureMaskingUtil.droppedFeaturesCount;
        }
        
        log.info("Pre-loading data to avoid the MemcpyAsync bug on GPU....");
        CrossValidationUtils[] temp = new CrossValidationUtils[nTasks];
        PreloadedCatanDataSetIterator[] preloadedIter = new PreloadedCatanDataSetIterator[nTasks];
        for(int j = 0; j < nTasks; j++) {
        	temp[j] = new CrossValidationUtils(fullIter[j], 10, nSamples);//only to get access to the full data
        	preloadedIter[j] = new PreloadedCatanDataSetIterator(temp[j].getFullData(), miniBatchSize);
        }
        
        //train over all the data from the other tasks
        log.info("Build model....");
        //remove the biases, since dl4j adds it's own and the iterator removes the bias from the data
        CatanMlpConfig modelConfig = new CatanMlpConfig(numInputs + actInputSize-2, 1, seed, iterations, WeightInit.XAVIER, Updater.RMSPROP, learningRate, LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        CatanMlp model = modelConfig.init();
        
        CatanPlotter prePlotter = new CatanPlotter(TASK);
        CatanEvaluation preEval = new CatanEvaluation(maxActions[TASK]);
        ActionGuessFeaturesEvaluation preAgfe = new ActionGuessFeaturesEvaluation(actInputSize-1);//get rid of the bias
        INDArray preOutput;
        
        
        log.info("Pretrain model on data from the other tasks....");
        for( int i=0; i<preEpochs; i++ ) {
        	//fitting one batch from each task repeatedly until we cover all samples for the pretraining
	        while (hasNext(preloadedIter)) {
	        	for(int j= 0; j < nTasks; j++){
	        		if(j==TASK)//do not train on the full data for the final task, since that will be cheating
	        			continue;
	        		if(!preloadedIter[j].hasNext())//skip task if we ran out of data
	        			continue;
	        		if(j == 4 && parser.getMaskHiddenFeatures())//we can't train on discard task when the input is masked since there is no difference between the actions
	        			continue;
            		CatanDataSet d = preloadedIter[j].next();
            		DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
                	if(NORMALISATION)
                		norm.normalizeZeroMeanUnitVariance(ds);
            		model.fit(ds,d.getSizeOfLegalActionSet());//set the labels we want to fit to
            	}
            }
            log.info("*** Completed pretraining epoch {} ***", i);
            
            log.info("Saving initial network parameters", i);
            ModelUtils.saveMlpModelAndParameters(model, 6);//all tasks together are represented as task 6
            
            //out of curiosity... how much does this training aids to generalise to the task we are really training?
            log.info("Evaluate model on training data for final task..");
            preEval = new CatanEvaluation(maxActions[TASK]);
            preAgfe = new ActionGuessFeaturesEvaluation(actInputSize-1);//get rid of the bias
            preloadedIter[TASK].reset();
            while (preloadedIter[TASK].hasNext()) {
            	CatanDataSet d = preloadedIter[TASK].next(1);
            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
            	preOutput = model.output(ds, d.getSizeOfLegalActionSet());
	            preEval.eval(DataUtils.computeLabels(d), preOutput.transposei());
	            preAgfe.eval(d.getTargetAction(), d.getActionFeatures().getRow(Nd4j.getBlasWrapper().iamax(preOutput)));
            }
            //once finished just output the result
            System.out.println(preEval.stats());
            prePlotter.addData(preEval.score(), preEval.score(), preEval.accuracy(), preEval.accuracy());
            prePlotter.addRanks(preEval.getRank(), preEval.getRank());
            prePlotter.addFeatureConfusion(preAgfe.getStats(), preAgfe.getStats());
            
            //reset iterators for future
            for(int j= 0; j < nTasks; j++){
            	preloadedIter[j].reset();
            }
            
        }
        //write current results and move files so these won't be overwritten
        prePlotter.writeAll();
        File dirFrom = new File(CatanPlotter.RESULTS_DIR);
        File[] files = dirFrom.listFiles();
        File dirTo = new File("preTrain");
        dirTo.mkdirs();
        for(File f : files){
        	FileUtils.moveFileToDirectory(f, dirTo, false);
        }
        
        //start training on the actual task now
        log.info("Build model using pretrained parameters....");
        //remove the biases, since dl4j adds it's own and the iterator removes the bias from the data
        CatanMlpConfig netConfig = new CatanMlpConfig(numInputs + actInputSize - 2, 1, seed, iterations, WeightInit.XAVIER, Updater.RMSPROP, learningRate, LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        CatanMlp network = netConfig.init();
        network.setParams(model.params());
    	
        CatanEvaluation tEval;
        ActionGuessFeaturesEvaluation tagfe;
        CatanPlotter plotter = new CatanPlotter(TASK);
        INDArray output;
        for( int i=0; i<epochs; i++ ) {
        	while(preloadedIter[TASK].hasNext()){
        		CatanDataSet d = preloadedIter[TASK].next();
        		DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
        		network.fit(ds,d.getSizeOfLegalActionSet());//set the labels we want to fit to
        	}
        	preloadedIter[TASK].reset();
            log.info("*** Completed epoch {} ***", i);
            
            log.info("Saving network parameters");
            ModelUtils.saveMlpModelAndParameters(network, TASK);
            
            log.info("Evaluate model on training set....");
            tEval = new CatanEvaluation(maxActions[TASK]);
            tagfe = new ActionGuessFeaturesEvaluation(actInputSize-1);//get rid of the bias
            while (preloadedIter[TASK].hasNext()) {
            	CatanDataSet d = preloadedIter[TASK].next(1);
            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
            	output = network.output(ds, d.getSizeOfLegalActionSet());
	            tEval.eval(DataUtils.computeLabels(d), output.transposei());
	            tagfe.eval(d.getTargetAction(), d.getActionFeatures().getRow(Nd4j.getBlasWrapper().iamax(output)));
            }
            preloadedIter[TASK].reset();
            //once finished just output the result
            System.out.println(tEval.stats());
            System.out.println("Feature confusion: " + Arrays.toString(tagfe.getStats()));
            plotter.addData(0, tEval.score(), 0, tEval.accuracy());
            plotter.addRanks(tEval.getRank(), tEval.getRank());
            plotter.addFeatureConfusion(tagfe.getStats(), tagfe.getStats());
            plotter.writeAll();
        }
    }  
	
	private static boolean hasNext(PreloadedCatanDataSetIterator[] fullIter){
        for(int j= 0; j < nTasks; j++){
    		if(j == 4 && parser.getMaskHiddenFeatures())
    			continue;
        	if(j!=TASK && fullIter[j].hasNext())
        		return true;
        }
		return false;	
	}
}
