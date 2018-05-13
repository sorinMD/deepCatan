package test.catan.logistic;

import java.io.File;
import java.util.ArrayList;
import java.util.Scanner;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.CatanEvaluation;
import org.deeplearning4j.eval.CatanPlotter;
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
import data.PreloadedCatanDataSetIterator;
import util.CatanFeatureMaskingUtil;
import util.CrossValidationUtils;
import util.DataUtils;
import util.NNConfigParser;

/**
 * Cross-validation version of the logistic regression baseline to train/test on human data for the whole dataset (i.e. all tasks so the single model).
 * 
 * @author sorinMD
 *
 */
public class CVLogRegFullDataSet {
	private static Logger log = LoggerFactory.getLogger(CVLogRegFullDataSet.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static NNConfigParser parser;
	private static boolean NORMALISATION;
	private static int nTasks = 6;
	private static boolean softmaxOverOut = true;
	private static boolean selectMax = false;

    public static void main(String[] args) throws Exception {
    	parser = new NNConfigParser(null);//TODO: add config file to resources and accept different ones via the args
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
        
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(iterations)
                .learningRate(learningRate)
                .rmsDecay(0.9)
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

        MultiLayerNetwork model;

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
	        
	        for(int i= 0; i < nTasks; i++){
		        trainIter[i] = cv[i].getTrainIterator();
	        	testIter[i] = cv[i].getTestIterator();
		        plotter[i][k] = new CatanPlotter(i);
		        eval[i] = new CatanEvaluation(trainMaxActions[i]);
		        tEval[i] = new CatanEvaluation(trainMaxActions[i]);
	        }
        	
        	//read all data for training into a datastructure
	        ArrayList<CatanDataSet>[] train = new ArrayList[6];
	        ArrayList<CatanDataSet>[] trainSamples = new ArrayList[6];
	        ArrayList<CatanDataSet>[] evalData = new ArrayList[6];
	        for(int j= 0; j < nTasks; j++){
	        	train[j] = new ArrayList<>(nSamples);
	        	while(trainIter[j].hasNext()){
	        		train[j].add(trainIter[j].next());
	        	}
	        	trainIter[j].reset();
	            //read all data for evaluation on training set
	            trainSamples[j] = new ArrayList<>(nSamples);
	        	while(trainIter[j].hasNext()){
	        		trainSamples[j].add(trainIter[j].next(1));
	        	}
	        	trainIter[j].reset();
	            //read all data for evaluation into a datastructure
	            evalData[j] = new ArrayList<>(nSamples);
	            while(testIter[j].hasNext()){
	        		evalData[j].add(testIter[j].next());
	        	}
	        	testIter[j].reset();
	        }
        	//reset model for each fold
            model = new MultiLayerNetwork(conf);
            model.init();
	        
            log.info("Train model....");
	        for( int i=0; i<epochs; i++ ) {
	        	for(int n = 0; n < train[2].size(); n++){
	        		for(int j= 0; j < nTasks; j++){
	        			if(train[j].size() <= n)
	        				continue;
		        		CatanDataSet d = train[j].get(n);
		        		DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
		            	if(NORMALISATION)
		            		norm.normalizeZeroMeanUnitVariance(ds);
		            	model.fit(ds);
	        		}
	        	}
	            log.info("*** Completed epoch {} ***", i);
	            
	            log.info("Evaluate model ....");
	            for(int j= 0; j < nTasks; j++){
	            	eval[j] = new CatanEvaluation(trainMaxActions[j]);
		            for(CatanDataSet d : evalData[j]){
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
//		            System.out.println(eval[j].stats());
		            
		            tEval[j] = new CatanEvaluation(trainMaxActions[j]);
		            for(CatanDataSet d : trainSamples[j]){
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
		            //once finished just output the result
//		            System.out.println(tEval[j].stats());
		            plotter[j][k].addData(eval[j].score(), tEval[j].score(), eval[j].accuracy(), tEval[j].accuracy());
		            plotter[j][k].addRanks(tEval[j].getRank(), eval[j].getRank());
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

