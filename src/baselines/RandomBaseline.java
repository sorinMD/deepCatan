package baselines;

import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;


/**
 * Takes in the number of actions and returns an array with the action index set to 1 and chosen at random.
 * 
 * @author sorinMD
 *
 */
public class RandomBaseline {
	Random rand;

	public RandomBaseline(int seed) {
		rand = new Random(seed);
	}

	/**
	 * 
	 * @param nActions
	 * @return
	 */
	public INDArray predict(int nActions){
		int index = rand.nextInt(nActions);
		INDArray ret = Nd4j.zeros(nActions);
		ret.get(NDArrayIndex.interval(index,index+1)).assign(1);
		return ret;
	}
	
	/**
	 * 
	 * @param nActions
	 * @return
	 */
	public int predictIndex(int nActions){
		 return rand.nextInt(nActions);
	}
	
}
