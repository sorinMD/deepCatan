package test.catan.logistic;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
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
 * Logistic regression baseline training with a fixed minibatch size for one task.
 * @author sorinMD
 *
 */
public class MBCatanLogisticRegression {
	private static Logger log = LoggerFactory.getLogger(MBCatanLogisticRegression.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static int TASK;
	private static NNConfigParser parser;
	private static boolean NORMALISATION;
	private static boolean softmaxOverOut = true;
	private static boolean selectMax = false;

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
        CatanDataSetIterator trainIter = new CatanDataSetIterator(trainData,nSamples,miniBatchSize,numInputs+1,numInputs+actInputSize+1,true,parser.getMaskHiddenFeatures());
        Normaliser norm = new Normaliser(PATH + DATA_TYPE,parser.getMaskHiddenFeatures());
        if(NORMALISATION){
        	log.info("Check normalisation parameters for dataset ....");
        	norm.init(trainIter, TASK);
        }
        //read all data for training into a datastructure
        ArrayList<CatanDataSet> train = new ArrayList<>(nSamples);
    	while(trainIter.hasNext()){
    		train.add(trainIter.next());
    	}
    	trainIter.reset();
    	
        //read all data for evaluation on training set
        ArrayList<CatanDataSet> trainSamples = new ArrayList<>(nSamples);
    	while(trainIter.hasNext()){
    		trainSamples.add(trainIter.next(1));
    	}
    	trainIter.reset();
        
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
        CatanDataSetIterator testIter = new CatanDataSetIterator(testData,10000,1,numInputs+1,numInputs+actInputSize+1,true,parser.getMaskHiddenFeatures());
        
        //read all data for evaluation into a datastructure
        ArrayList<CatanDataSet> evalData = new ArrayList<>(nSamples);
        while(testIter.hasNext()){
    		evalData.add(testIter.next());
    	}
    	testIter.reset();
    	
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

        //used to write results to file
        CatanPlotter plotter = new CatanPlotter(parser.getTask());
        
        int mb = 10;//TODO: find a way to include this in the configuration
        
        log.info("Train model....");
        for( int i=0; i<epochs; i++ ) {
        	trainIter.reset();
        	DataSet currentDs = DataUtils.turnToSAPairsDS(trainIter.next(),metricW,labelW);
        	int idx = 0;
        	while(trainIter.hasNext()){
            	List<INDArray> inputs = new ArrayList<>();
                List<INDArray> labels = new ArrayList<>();
        		for(int k = 0; k < mb; k++){//create the new dataset
        			if(idx == currentDs.getFeatureMatrix().size(0)){
        				if(trainIter.hasNext()){
        					currentDs = DataUtils.turnToSAPairsDS(trainIter.next(),metricW,labelW);
        					idx = 0;
        				}else
        					break;
        			}
        			inputs.add(currentDs.getFeatureMatrix().getRow(idx));
        			labels.add(currentDs.getLabels().getRow(idx));
        			idx++;
        		}
        		DataSet ds = new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])), Nd4j.vstack(labels.toArray(new INDArray[0])));
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
        		model.fit(ds);
        	}
            log.info("*** Completed epoch {} ***", i);
            
            CatanEvaluation eval = new CatanEvaluation(maxActions);
            CatanEvaluation tEval = new CatanEvaluation(trainMaxActions);
            
            log.info("Evaluate model on evaluation set....");
            for(CatanDataSet d : evalData){
            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
            	if(softmaxOverOut){
	            	INDArray output = model.output(ds.getFeatureMatrix());
	    	    	SoftMax softMax = new SoftMax(output);
	    	    	softMax.exec(1);
	    	    	if(selectMax){
	    	    		int k = Nd4j.getBlasWrapper().iamax(softMax.z().getColumn(0));
	    	    		double[] out = new double[d.getSizeOfLegalActionSet().getInt(0)];
	    	    		out[k] = 1;
	    	    		eval.eval(DataUtils.computeLabels(d), Nd4j.create(out));
	    	    	}else{//let the evaluation to select the max
	    	    		eval.eval(DataUtils.computeLabels(d), softMax.z().transpose());
	    	    	}
            	}else{
            		INDArray output = model.output(ds.getFeatureMatrix());
	    	    	if(selectMax){
	    	    		int k = Nd4j.getBlasWrapper().iamax(output.getColumn(0));
	    	    		double[] out = new double[d.getSizeOfLegalActionSet().getInt(0)];
	    	    		out[k] = 1;
	    	    		eval.eval(DataUtils.computeLabels(d), Nd4j.create(out));
	    	    	}else{//let the evaluation to select the max
	    	    		eval.eval(DataUtils.computeLabels(d), output.transpose());
	    	    	}
            	}
            }
            //once finished just output the result
            System.out.println(eval.stats());
            
            log.info("Evaluate model on training set....");
            for(CatanDataSet d : trainSamples){
            	DataSet ds = DataUtils.turnToSAPairsDS(d,metricW,labelW);
            	if(NORMALISATION)
            		norm.normalizeZeroMeanUnitVariance(ds);
            	if(softmaxOverOut){
	            	INDArray output = model.output(ds.getFeatureMatrix());
	    	    	SoftMax softMax = new SoftMax(output);
	    	    	softMax.exec(1);
	    	    	if(selectMax){
	    	    		int k = Nd4j.getBlasWrapper().iamax(softMax.z().getColumn(0));
	    	    		double[] out = new double[d.getSizeOfLegalActionSet().getInt(0)];
	    	    		out [k] = 1;
	    	    		tEval.eval(DataUtils.computeLabels(d), Nd4j.create(out));
	    	    	}else{
	    	    		tEval.eval(DataUtils.computeLabels(d), softMax.z().transpose());
	    	    	}
            	}else{
	            	INDArray output = model.output(ds.getFeatureMatrix());
	            	if(selectMax){
	    	    		int k = Nd4j.getBlasWrapper().iamax(output.getColumn(0));
	    	    		double[] out = new double[d.getSizeOfLegalActionSet().getInt(0)];
	    	    		out[k] = 1;
	    	    		tEval.eval(DataUtils.computeLabels(d), Nd4j.create(out));
	    	    	}else{//let the evaluation to select the max
	    	    		tEval.eval(DataUtils.computeLabels(d), output.transpose());
	    	    	}
	            }
            }
            //once finished just output the result
            System.out.println(tEval.stats());
            
            plotter.addData(eval.score(), tEval.score(), eval.accuracy(), tEval.accuracy());
            plotter.addRanks(tEval.getRank(), eval.getRank());
//            plotter.plotAll();
            plotter.writeAll();
            
        
        }

    }
	
	
}
