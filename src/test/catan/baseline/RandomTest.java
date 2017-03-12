package test.catan.baseline;

import java.io.File;
import java.util.Scanner;

import org.deeplearning4j.eval.CatanEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import baselines.RandomBaseline;
import data.CatanDataSet;
import data.CatanDataSetIterator;
import util.DataUtils;
import util.NNConfigParser;

/**
 * Random baseline. This approach is too weak to be used as an actual baseline.
 * 
 * @author sorinMD
 *
 */
public class RandomTest {

	private static Logger log = LoggerFactory.getLogger(RandomTest.class);
	private static final String METADATA_SEPRATOR = ":";
	private static final String PATH = "/afs/inf.ed.ac.uk/group/project/stac_data/data/";
	private static String DATA_TYPE;
	private static int task;
	private static NNConfigParser parser;
	
	public static void main(String[] args) throws Exception {
    	parser = new NNConfigParser(null);
    	task = parser.getTask();
    	DATA_TYPE = parser.getDataType();
		
		log.info("Build model....");
		RandomBaseline model = new RandomBaseline(123);
	
        //get the test data from completely different games
        log.info("Load test data....");
        File testData = new File(PATH + DATA_TYPE + "/test-" + task + ".txt");
    	Scanner testScanner = new Scanner(new File(PATH + DATA_TYPE + "/test-" + task + "-metadata.txt"));
    	if(!testScanner.hasNextLine()){
    		testScanner.close();
    		throw new RuntimeException("Metadata not found; Cannot initialise evaluation set");
    	}
    	int nSamples = parser.getNumberOfSamples();
    	int numInputs = Integer.parseInt(testScanner.nextLine().split(METADATA_SEPRATOR)[1]);
        int actInputSize = Integer.parseInt(testScanner.nextLine().split(METADATA_SEPRATOR)[1]);
        int maxActions = Integer.parseInt(testScanner.nextLine().split(METADATA_SEPRATOR)[1]);
        testScanner.close();
        CatanDataSetIterator testIter = new CatanDataSetIterator(testData,nSamples,1,numInputs+1,numInputs+actInputSize+1,true);
        
        CatanEvaluation eval = new CatanEvaluation(maxActions);
        log.info("Evaluate model....");
        while (testIter.hasNext()) {
        	CatanDataSet d = testIter.next();
        	eval.eval(DataUtils.computeLabels(d), model.predict(d.getSizeOfLegalActionSet().getInt(0)));
        }
        System.out.println(eval.stats());
	}

}
