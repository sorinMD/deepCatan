package test.catan.baseline;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import org.deeplearning4j.eval.CatanEvaluation;
import org.deeplearning4j.eval.CatanPlotter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import data.CatanDataSet;
import data.CatanDataSetIterator;
import util.DataUtils;
import util.Key;
import util.NNConfigParser;

/**
 * The frequency based baseline.
 * 
 * @author sorinMD
 *
 */
public class FrequencyBased {
	private static Logger log = LoggerFactory.getLogger(FrequencyBased.class);
	private static final String METADATA_SEPARATOR = ":";
	private static String PATH;
	private static String DATA_TYPE;
	private static int TASK;
	private static NNConfigParser parser;
	
	public static void main(String[] args) throws Exception {
    	parser = new NNConfigParser(null);
    	TASK = parser.getTask();
    	DATA_TYPE = parser.getDataType();
    	PATH = parser.getDataPath();
        double labelW = parser.getLabelWeight();
        double metricW = parser.getMetricWeight();
        
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
        
        //data iterator
        CatanDataSetIterator trainIter = new CatanDataSetIterator(trainData,nSamples,1,numInputs+1,numInputs+actInputSize+1,true,parser.getMaskHiddenFeatures());
        
//        //read all data for training into a datastructure
//        ArrayList<CatanDataSet> train = new ArrayList<>(nSamples);
//    	while(trainIter.hasNext()){
//    		train.add(trainIter.next(1));
//    	}
//    	trainIter.reset();
    	
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
        
//        //read all data for evaluation into a datastructure
//        ArrayList<CatanDataSet> evalData = new ArrayList<>(nSamples);
//        while(testIter.hasNext()){
//    		evalData.add(testIter.next());
//    	}
//    	testIter.reset();
    	
        //used to write results to file
        CatanPlotter plotter = new CatanPlotter(parser.getTask());
        
        //get the count of each action based on the features
        Map<Key, Integer> actionCount = new HashMap<>();
        while(trainIter.hasNext()){
        	CatanDataSet cds = trainIter.next();
        	DataSet ds = DataUtils.turnToSAPairsDS(cds,metricW,labelW);
        	int idx = Nd4j.getBlasWrapper().iamax(ds.getLabels().getColumn(0));
        	
        	Key key = new Key(cds.getActionFeatures().getRow(idx)); 
        	if(!actionCount.containsKey(key))
        		actionCount.put(key, 1);
        	else{
        		Integer val = actionCount.get(key);
        		val++;
        		actionCount.put(key, val);
        	}
        }
        trainIter.reset();

        CatanEvaluation eval = new CatanEvaluation(maxActions);
        CatanEvaluation tEval = new CatanEvaluation(trainMaxActions);
        Random rnd = new Random();
        double eps = 1e-6;
        
        log.info("Evaluate model on evaluation set....");
        while(testIter.hasNext()){
        	CatanDataSet d = testIter.next();
        	double maxVal = 0;
        	int idx = 0;
        	for (int i = 0; i < d.getSizeOfLegalActionSet().getInt(0); i++){
        		Key k = new Key(d.getActionFeatures().getRow(i));
        		double val = 0;
        		if(actionCount.containsKey(k)){
        			val = (double)actionCount.get(k) + rnd.nextDouble() * eps;
        		}
        		if(val > maxVal){
        			idx = i;
        			maxVal = i;
        		}
        	}
        	
        	double[] result = new double[d.getSizeOfLegalActionSet().getInt(0)];
        	result[idx] = 1;
        	INDArray out = Nd4j.create(result);
    	    eval.eval(DataUtils.computeLabels(d), out.transpose());
        }
        //once finished just output the result
        System.out.println(eval.stats());
        
        
        log.info("Evaluate model on train set....");
        while(trainIter.hasNext()){
        	CatanDataSet d = trainIter.next();
        	double maxVal = 0;
        	int idx = 0;
        	for (int i = 0; i < d.getSizeOfLegalActionSet().getInt(0); i++){
        		Key k = new Key(d.getActionFeatures().getRow(i));
        		double val = 0;
        		if(actionCount.containsKey(k)){
        			val = (double)actionCount.get(k) + rnd.nextDouble() * eps;
        		}
        		if(val > maxVal){
        			idx = i;
        			maxVal = i;
        		}
        	}
        	
        	double[] result = new double[d.getSizeOfLegalActionSet().getInt(0)];
        	result[idx] = 1;
        	INDArray out = Nd4j.create(result);
    	    tEval.eval(DataUtils.computeLabels(d), out.transpose());
        }
        //once finished just output the result
        System.out.println(tEval.stats());              

    }

}
