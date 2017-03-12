package org.deeplearning4j.eval;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Evaluator that can be used to analyse which features is the agent most likely to get wrong.
 * It achieves this by counting when it was correct/incorrect. This should provide information on what type of action is likely to get wrong.
 * e.g did it guess the location of the robber and got the player wrong or the other way around?
 * 
 * @author sorinMD
 *
 */
public class ActionGuessFeaturesEvaluation {
	/**
	 * Used to keep track of the number of times these were correct;
	 */
    private int[] incorrectCounter;
    private int samples = 0;
    
    public ActionGuessFeaturesEvaluation(int nFeatures) {
    	incorrectCounter = new int[nFeatures];
    }
    
    public void eval(INDArray correct,INDArray guess) {
    	samples++;
    	if(!correct.equals(guess)){
    		INDArray diff = correct.sub(guess);
    		for(int i=0; i< incorrectCounter.length; i++){
    			if(diff.getColumn(i).getDouble(0) > 0)
    				incorrectCounter[i]++;
    		}
    	}
    }
    
    public double[] getStats() {
    	double[] stats = new double[incorrectCounter.length];
    	for(int i=0 ; i < stats.length ; i++){
    		stats[i] = (double)incorrectCounter[i];
    	}
        return stats;
    }
    
    // Other misc methods
    public void reset(){
    	samples = 0;
    	incorrectCounter = new int[incorrectCounter.length];
    }
}
