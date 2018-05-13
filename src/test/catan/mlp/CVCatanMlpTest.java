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
import util.CatanFeatureMaskingUtil;
import util.CrossValidationUtils;
import util.DataUtils;
import util.NNConfigParser;

/**
 * Training and testing the CatanMlp model using cross-validation.
 * 
 * @author sorinMD
 *
 */
public class CVCatanMlpTest {
	private static Logger log = LoggerFactory.getLogger(CVCatanMlpTest.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static int TASK;
	private static NNConfigParser parser;
	private static boolean NORMALISATION;
	
	 public static void main(String[] args) throws Exception {
	    	parser = new NNConfigParser(null);
	    	TASK = parser.getTask();
	    	DATA_TYPE = parser.getDataType();
	    	DATA_TYPE = DATA_TYPE + "CV";
	    	PATH = parser.getDataPath();
	    	NORMALISATION = parser.getNormalisation();
	    	int FOLDS = 10; //standard 10 fold validation
	    	
	    	log.info("Load data....");
	    	//the file containing the training data
	    	File data = new File(PATH + DATA_TYPE + "/alldata-" + TASK + ".txt");
	        //read metadata and feed in all the required info
	    	Scanner scanner = new Scanner(new File(PATH + DATA_TYPE + "/alldata-" + TASK + "-metadata.txt"));
	    	if(!scanner.hasNextLine()){
	    		scanner.close();
	    		throw new RuntimeException("Metadata not found; Cannot initialise network parameters");
	    	}
	    	
	    	int numInputs = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
	        int actInputSize = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
	        int maxActions = Integer.parseInt(scanner.nextLine().split(METADATA_SEPARATOR)[1]);
	        int nSamples = parser.getNumberOfSamples();
	        scanner.close();
	        
	        //Set specific params
	        int epochs = parser.getEpochs();
	        int miniBatchSize = parser.getMiniBatchSize();
	        double learningRate = parser.getLearningRate();
	        //label softening
	        double labelW = parser.getLabelWeight();
	        double metricW = parser.getMetricWeight();
	            	
	        int iterations = 1; //iterations over each batch; this should always be equal to 1
	        long seed = 123;
	       
	        //full data iterator for data normalisation so we won't have to wait for every fold
	        CatanDataSetIterator fullIter = new CatanDataSetIterator(data,nSamples,miniBatchSize,numInputs+1,numInputs+actInputSize+1,true,parser.getMaskHiddenFeatures());
	        Normaliser norm = new Normaliser(PATH + DATA_TYPE,parser.getMaskHiddenFeatures());
	        if(NORMALISATION){
	        	log.info("Check normalisation parameters for dataset ....");
	        	norm.init(fullIter, TASK);
	        }
	        
	        log.info("Initialise cross-validation....");
	        CrossValidationUtils cv = new CrossValidationUtils(fullIter, FOLDS, nSamples);
	        CatanPlotter[] plotter = new CatanPlotter[FOLDS];
	        
	        //if the input is masked/postprocessed update the input size for the model creation
	        if(parser.getMaskHiddenFeatures()) {
	        	numInputs -= CatanFeatureMaskingUtil.droppedFeaturesCount;
	        	actInputSize -= CatanFeatureMaskingUtil.droppedFeaturesCount;
	        }
	        
	        for(int k = 0; k < FOLDS; k++){
	        	log.info("Starting fold " + k);
	        	cv.setCurrentFold(k);
	        	PreloadedCatanDataSetIterator trainIter = cv.getTrainIterator();
	        	PreloadedCatanDataSetIterator testIter = cv.getTestIterator();
	        	
		        //remove the biases, since dl4j adds it's own and the iterator removes the bias from the data
		        CatanMlpConfig netConfig = new CatanMlpConfig(numInputs + actInputSize - 2, 1, seed, iterations, WeightInit.XAVIER, Updater.RMSPROP, learningRate, LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		        CatanMlp network = netConfig.init();
	        	
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
		        		network.fit(ds,d.getSizeOfLegalActionSet());
		        	}
		        	trainIter.reset();
		            log.info("*** Completed epoch {} ***", i);
		            
		            log.info("Evaluate model on evaluation set....");
		            eval = new CatanEvaluation(maxActions);
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
//		            System.out.println("Feature confusion: " + Arrays.toString(agfe.getStats()));
		            
		            log.info("Evaluate model on training set....");
		            tEval = new CatanEvaluation(maxActions);
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
//		            System.out.println("Feature confusion: " + Arrays.toString(tagfe.getStats()));
		            plotter[k].addData(eval.score(), tEval.score(), eval.accuracy(), tEval.accuracy());
		            plotter[k].addRanks(tEval.getRank(), eval.getRank());
		            plotter[k].addFeatureConfusion(agfe.getStats(), tagfe.getStats());
		        }
		        
	            //write current results and move files so these won't be overwritten
		        //TODO: clean the results from the previous run
		        plotter[k].writeAll();
	            File dirFrom = new File(CatanPlotter.RESULTS_DIR);
	            File[] files = dirFrom.listFiles();
	            File dirTo = new File("" + k);
	            dirTo.mkdirs();
	            for(File f : files){
	            	FileUtils.moveFileToDirectory(f, dirTo, false);
	            }
	        }
	        cv.writeResults(plotter, TASK);
	 }
	 
}
