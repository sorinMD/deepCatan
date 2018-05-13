package test.catan.logistic;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Scanner;

import org.deeplearning4j.eval.CatanEvaluation;
import org.deeplearning4j.eval.CatanPlotter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import data.CatanDataSet;
import data.CatanDataSetIterator;
import data.Normaliser;
import util.CatanFeatureMaskingUtil;
import util.DataUtils;
import util.NNConfigParser;

/**
 * Logistic regression baseline train/test on any data type for the full dataset (i.e. all tasks so the single model).
 * 
 * @author sorinMD
 *
 */
public class CatanLogRegFullDataSet {
	private static Logger log = LoggerFactory.getLogger(CatanLogRegFullDataSet.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static int TASK;
	private static NNConfigParser parser;
	private static int nTasks = 6;
	private static boolean NORMALISATION;
	private static boolean softmaxOverOut = true;
	private static boolean selectMax = false;

    public static void main(String[] args) throws Exception {
    	parser = new NNConfigParser(null);//TODO: add config file to resources and accept different ones via the args
    	TASK = parser.getTask();
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
	        trainIter[i] = new CatanDataSetIterator(trainData[i],nSamples,miniBatchSize,numInputs+1,numInputs+actInputSize+1,true,parser.getMaskHiddenFeatures());
        }
        
        Normaliser norm = new Normaliser(PATH + DATA_TYPE,parser.getMaskHiddenFeatures());
        if(NORMALISATION){
        	log.info("Check normalisation parameters for dataset ....");
        	norm.init(trainIter);
        }

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
	        testIter[i] = new CatanDataSetIterator(testData[i],10000,1,numInputs+1,numInputs+actInputSize+1,true,parser.getMaskHiddenFeatures());
	        plotter[i] = new CatanPlotter(i);
	        eval[i] = new CatanEvaluation(trainMaxActions[i]);
	        tEval[i] = new CatanEvaluation(maxActions[i]);
        }
        
    	//read all data for training into a datastructure //NOTE: cannot do this on gpu for some reason...
//        ArrayList<CatanDataSet>[] train = new ArrayList[6];
//        ArrayList<CatanDataSet>[] trainSamples = new ArrayList[6];
//        ArrayList<CatanDataSet>[] evalData = new ArrayList[6];
//        for(int j= 0; j < nTasks; j++){
//        	train[j] = new ArrayList<>(nSamples);
//        	while(trainIter[j].hasNext()){
//        		train[j].add(trainIter[j].next());
//        	}
//        	trainIter[j].reset();
//            //read all data for evaluation on training set
//            trainSamples[j] = new ArrayList<>(nSamples);
//        	while(trainIter[j].hasNext()){
//        		trainSamples[j].add(trainIter[j].next(1));
//        	}
//        	trainIter[j].reset();
//            //read all data for evaluation into a datastructure
//            evalData[j] = new ArrayList<>(nSamples);
//            while(testIter[j].hasNext()){
//        		evalData[j].add(testIter[j].next());
//        	}
//        	testIter[j].reset();
//        }
    	
        //if the input is masked/postprocessed update the input size for the model creation
        if(parser.getMaskHiddenFeatures()) {
        	numInputs -= CatanFeatureMaskingUtil.droppedFeaturesCount;
        	actInputSize -= CatanFeatureMaskingUtil.droppedFeaturesCount;
        }
        
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(iterations)
                .learningRate(learningRate)
                .rmsDecay(0.9)
//                .l2(0.01)
//                .regularization(true)
                .list(2)
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs + actInputSize - 2)
                        .nOut(256)
                        .activation("sigmoid")
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.RMSPROP)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(256)
                        .nOut(256)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.RMSPROP)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunction.XENT)
                        .nIn(256)
                        .nOut(1)
                        .activation("sigmoid")
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.RMSPROP)
                        .build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model....");
        for( int i=0; i<epochs; i++ ) {
        	while (trainIter[0].hasNext() && trainIter[1].hasNext() && trainIter[2].hasNext() && trainIter[3].hasNext() && trainIter[4].hasNext() && trainIter[5].hasNext()) { //synth data where we need to keep them the same size
        		for(int j= 0; j < nTasks; j++){
	        		if(!trainIter[j].hasNext())
	        			continue;
	        		CatanDataSet d = trainIter[j].next();
	        		DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
	            	if(NORMALISATION)
	            		norm.normalizeZeroMeanUnitVariance(ds);
	            	model.fit(ds);
        		}
        	}
            log.info("*** Completed epoch {} ***", i);
            
            log.info("Evaluate model ....");
            for(int j= 0; j < nTasks; j++){
            	testIter[j].reset();
            	eval[j] = new CatanEvaluation(trainMaxActions[j]);
	            while (testIter[j].hasNext()) {
	            	CatanDataSet d = testIter[j].next();
	            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
	            	if(NORMALISATION)
	            		norm.normalizeZeroMeanUnitVariance(ds);
	            	if(softmaxOverOut){
		            	INDArray output = model.output(ds.getFeatureMatrix());
		    	    	SoftMax softMax = new SoftMax(output);
		    	    	softMax.exec(1);
		    	    	if(selectMax){
		    	    		int n = Nd4j.getBlasWrapper().iamax(softMax.z().getColumn(0));
		    	    		double[] out = new double[d.getSizeOfLegalActionSet().getInt(0)];
		    	    		out[n] = 1;
		    	    		eval[j].eval(DataUtils.computeLabels(d), Nd4j.create(out));
		    	    	}else{//let the evaluation to select the max
		    	    		eval[j].eval(DataUtils.computeLabels(d), softMax.z().transpose());
		    	    	}
	            	}else{
	            		INDArray output = model.output(ds.getFeatureMatrix());
		    	    	if(selectMax){
		    	    		int n = Nd4j.getBlasWrapper().iamax(output.getColumn(0));
		    	    		double[] out = new double[d.getSizeOfLegalActionSet().getInt(0)];
		    	    		out[n] = 1;
		    	    		eval[j].eval(DataUtils.computeLabels(d), Nd4j.create(out));
		    	    	}else{//let the evaluation to select the max
		    	    		eval[j].eval(DataUtils.computeLabels(d), output.transpose());
		    	    	}
	            	}
	            }
	            //once finished just output the result
//	            System.out.println(eval[j].stats());
	            trainIter[j].reset();
	            tEval[j] = new CatanEvaluation(trainMaxActions[j]);
	            while (trainIter[j].hasNext()) {
	            	CatanDataSet d = trainIter[j].next(1);
	            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
	            	if(NORMALISATION)
	            		norm.normalizeZeroMeanUnitVariance(ds);
	            	if(softmaxOverOut){
		            	INDArray output = model.output(ds.getFeatureMatrix());
		    	    	SoftMax softMax = new SoftMax(output);
		    	    	softMax.exec(1);
		    	    	if(selectMax){
		    	    		int n = Nd4j.getBlasWrapper().iamax(softMax.z().getColumn(0));
		    	    		double[] out = new double[d.getSizeOfLegalActionSet().getInt(0)];
		    	    		out [n] = 1;
		    	    		tEval[j].eval(DataUtils.computeLabels(d), Nd4j.create(out));
		    	    	}else{
		    	    		tEval[j].eval(DataUtils.computeLabels(d), softMax.z().transpose());
		    	    	}
	            	}else{
		            	INDArray output = model.output(ds.getFeatureMatrix());
		            	if(selectMax){
		    	    		int n = Nd4j.getBlasWrapper().iamax(output.getColumn(0));
		    	    		double[] out = new double[d.getSizeOfLegalActionSet().getInt(0)];
		    	    		out[n] = 1;
		    	    		tEval[j].eval(DataUtils.computeLabels(d), Nd4j.create(out));
		    	    	}else{//let the evaluation to select the max
		    	    		tEval[j].eval(DataUtils.computeLabels(d), output.transpose());
		    	    	}
		            }
	            }
	            trainIter[j].reset();
	            plotter[j].addData(eval[j].score(), tEval[j].score(), eval[j].accuracy(), tEval[j].accuracy());
	            plotter[j].addRanks(tEval[j].getRank(), eval[j].getRank());
//            plotter.plotAll();
	            plotter[j].writeAll();
            }
        }
    }
}
