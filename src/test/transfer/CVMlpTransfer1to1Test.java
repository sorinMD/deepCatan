package test.transfer;

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
import util.CrossValidationUtils;
import util.DataUtils;
import util.ModelUtils;
import util.NNConfigParser;

/**
 * Transfer learning test where the model is pre-trained on a single task before
 * being fine-tuned to the target task.
 * 
 * @author sorinMD
 */
public class CVMlpTransfer1to1Test {
	private static Logger log = LoggerFactory.getLogger(CVMlpTransferTest.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static NNConfigParser parser;
	private static int TASK;
	private static boolean NORMALISATION;
	/**
	 * The number of epochs the system is trained on all the data from the tasks we are trying to transfer from
	 */
	private static int preEpochs = 1;
	
	public static void main(String[] args) throws Exception {
    	parser = new NNConfigParser(null);
    	DATA_TYPE = parser.getDataType();
    	DATA_TYPE = DATA_TYPE + "CV";
    	PATH = parser.getDataPath();
    	TASK = parser.getTask();
    	NORMALISATION = parser.getNormalisation();
    	int FOLDS = 10; //standard 10 fold validation
    	int PRETASK = parser.getPreTrainTask();
    	
    	//the features used are the same across the tasks so we just use the final ones, but we need to read them as they are present in all metadata files
    	int numInputs = 0;
        int actInputSize = 0;
        int[] maxActions = new int[2];//this is different for each task;
    	
    	log.info("Load data for the two tasks....");
    	File[] data = new File[2];
		data[0] = new File(PATH + DATA_TYPE + "/alldata-" + PRETASK + ".txt");
        //read metadata and feed in all the required info
    	Scanner scanner = new Scanner(new File(PATH + DATA_TYPE + "/alldata-" + PRETASK + "-metadata.txt"));
    	if(!scanner.hasNextLine()){
    		scanner.close();
    		throw new RuntimeException("Metadata not found; Cannot initialise network parameters");
    	}
    	numInputs = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
        actInputSize = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
        maxActions[0] = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
        scanner.close();
    	
		data[1] = new File(PATH + DATA_TYPE + "/alldata-" + TASK + ".txt");
        //read metadata and feed in all the required info
    	scanner = new Scanner(new File(PATH + DATA_TYPE + "/alldata-" + TASK + "-metadata.txt"));
    	if(!scanner.hasNextLine()){
    		scanner.close();
    		throw new RuntimeException("Metadata not found; Cannot initialise network parameters");
    	}
    	numInputs = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
        actInputSize = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
        maxActions[1] = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
        scanner.close();
        
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
        log.info("Generate the iterators for the two tasks only.");
        CatanDataSetIterator[] fullIter = new CatanDataSetIterator[2];
        fullIter[0] = new CatanDataSetIterator(data[0],nSamples,miniBatchSize,numInputs+1,numInputs+actInputSize+1,true);
        fullIter[1] = new CatanDataSetIterator(data[1],nSamples,miniBatchSize,numInputs+1,numInputs+actInputSize+1,true);
        
        Normaliser norm = new Normaliser(PATH + DATA_TYPE);
        if(NORMALISATION){
        	log.info("Check normalisation parameters for dataset ....");
        	norm.init(fullIter,TASK,PRETASK);
        }
        
        //train over the task chosen for pre-training
        log.info("Build model....");
        //remove the biases, since dl4j adds it's own and the iterator removes the bias from the data
        CatanMlpConfig modelConfig = new CatanMlpConfig(numInputs + actInputSize-2, 1, seed, iterations, WeightInit.XAVIER, Updater.RMSPROP, learningRate, LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        CatanMlp model = modelConfig.init();
        
        CatanPlotter prePlotter = new CatanPlotter(TASK);
        CatanEvaluation preEval = new CatanEvaluation(maxActions[1]);
        ActionGuessFeaturesEvaluation preAgfe = new ActionGuessFeaturesEvaluation(actInputSize-1);//get rid of the bias
        INDArray preOutput;
        
        log.info("Pretrain model on data from the other tasks....");
        for( int i=0; i<preEpochs; i++ ) {
        	//fitting one batch from each task repeatedly until we cover all samples
	        while (fullIter[0].hasNext()) { //task 2 always has the most data
				CatanDataSet d = fullIter[0].next();
				DataSet ds = DataUtils.turnToSAPairsDS(d, metricW, labelW);
				if (NORMALISATION)
					norm.normalizeZeroMeanUnitVariance(ds);
				model.fit(ds,d.getSizeOfLegalActionSet());//set the labels we want to fit to
            }
            log.info("*** Completed epoch {} ***", i);
            
            log.info("Saving network parameters", i);
            ModelUtils.saveMlpModelAndParameters(model, 6);//all tasks together are represented as task 6
            
            //out of curiosity... how much does this training aids to generalise to the task we are really training?
            log.info("Evaluate model on training data for final task..");
            preEval = new CatanEvaluation(maxActions[1]);
            preAgfe = new ActionGuessFeaturesEvaluation(actInputSize-1);//get rid of the bias
            fullIter[1].reset();
            while (fullIter[1].hasNext()) {
            	CatanDataSet d = fullIter[1].next(1);
            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
            	preOutput = model.output(ds, d.getSizeOfLegalActionSet());
	            preEval.eval(DataUtils.computeLabels(d), preOutput.transposei());
	            preAgfe.eval(d.getTargetAction(), d.getActionFeatures().getRow(Nd4j.getBlasWrapper().iamax(preOutput)));
            }
            fullIter[1].reset();
            //once finished just output the result
            System.out.println(preEval.stats());
            prePlotter.addData(preEval.score(), preEval.score(), preEval.accuracy(), preEval.accuracy());
            prePlotter.addRanks(preEval.getRank(), preEval.getRank());
            prePlotter.addFeatureConfusion(preAgfe.getStats(), preAgfe.getStats());
            
            //reset iterators
            fullIter[0].reset();
            fullIter[1].reset();
            
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
        log.info("Initialise cross-validation....");
        CrossValidationUtils cv = new CrossValidationUtils(fullIter[1], FOLDS, nSamples);
        
        CatanPlotter[] plotter = new CatanPlotter[FOLDS];
        
        for(int k = 0; k < FOLDS; k++){
        	log.info("Starting fold " + k);
        	cv.setCurrentFold(k);
        	PreloadedCatanDataSetIterator trainIter = cv.getTrainIterator();
        	PreloadedCatanDataSetIterator testIter = cv.getTestIterator();
        
	        log.info("Build model....");
	        //remove the biases, since dl4j adds it's own and the iterator removes the bias from the data
	        CatanMlpConfig netConfig = new CatanMlpConfig(numInputs + actInputSize - 2, 1, seed, iterations, WeightInit.XAVIER, Updater.RMSPROP, learningRate, LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
	        CatanMlp network = netConfig.init();
	        network.setParams(model.params());
        	
	        plotter[k] = new CatanPlotter(parser.getTask());
	        CatanEvaluation eval;
	        CatanEvaluation tEval;
	        ActionGuessFeaturesEvaluation tagfe;
	        ActionGuessFeaturesEvaluation agfe;
	        
	        INDArray output;
	        log.info("Train/evaluate model on fold " + k);
	        for( int i=0; i<epochs; i++ ) {
	        	while(trainIter.hasNext()){
	        		CatanDataSet d = trainIter.next();
	        		DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
	            	if(NORMALISATION)
	            		norm.normalizeZeroMeanUnitVariance(ds);
	        		network.fit(ds,d.getSizeOfLegalActionSet());//set the labels we want to fit to
	        	}
	        	trainIter.reset();
	            log.info("*** Completed epoch {} ***", i);
	            
	            log.info("Evaluate model on evaluation set....");
	            eval = new CatanEvaluation(maxActions[1]);
	            agfe = new ActionGuessFeaturesEvaluation(actInputSize-1);//get rid of the bias
	            while (testIter.hasNext()) {
	            	CatanDataSet d = testIter.next();
	            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
	            	if(NORMALISATION)
	            		norm.normalizeZeroMeanUnitVariance(ds);
	            	output = network.output(ds, d.getSizeOfLegalActionSet());
		            eval.eval(DataUtils.computeLabels(d), output.transposei());
		            agfe.eval(d.getTargetAction(), d.getActionFeatures().getRow(Nd4j.getBlasWrapper().iamax(output)));
	            }
	            testIter.reset();
	            //once finished just output the result
	            System.out.println(eval.stats());
	            System.out.println("Feature confusion: " + Arrays.toString(agfe.getStats()));
	            
	            log.info("Evaluate model on training set....");
	            tEval = new CatanEvaluation(maxActions[1]);
	            tagfe = new ActionGuessFeaturesEvaluation(actInputSize-1);//get rid of the bias
	            while (trainIter.hasNext()) {
	            	CatanDataSet d = trainIter.next(1);
	            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
	            	if(NORMALISATION)
	            		norm.normalizeZeroMeanUnitVariance(ds);
	            	output = network.output(ds, d.getSizeOfLegalActionSet());
		            tEval.eval(DataUtils.computeLabels(d), output.transposei());
		            tagfe.eval(d.getTargetAction(), d.getActionFeatures().getRow(Nd4j.getBlasWrapper().iamax(output)));
	            }
	            trainIter.reset();
	            //once finished just output the result
	            System.out.println(tEval.stats());
	            System.out.println("Feature confusion: " + Arrays.toString(tagfe.getStats()));
	            plotter[k].addData(eval.score(), tEval.score(), eval.accuracy(), tEval.accuracy());
	            plotter[k].addRanks(tEval.getRank(), eval.getRank());
	            plotter[k].addFeatureConfusion(agfe.getStats(), tagfe.getStats());
	        }
	        
            //write current results and move files so these won't be overwritten
	        plotter[k].writeAll();
            dirFrom = new File(CatanPlotter.RESULTS_DIR);
            files = dirFrom.listFiles();
            dirTo = new File("" + k);
            dirTo.mkdirs();
            for(File f : files){
            	FileUtils.moveFileToDirectory(f, dirTo, false);
            }
	        
        }
        cv.writeResults(plotter, TASK);
        
    }  
	
}
