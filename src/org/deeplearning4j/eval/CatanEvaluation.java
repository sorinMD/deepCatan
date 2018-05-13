package org.deeplearning4j.eval;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossCalculation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Evaluation metrics: accuracy, the position of the true option in the network's prediction
 * @author sorinMD
 *
 */
public class CatanEvaluation implements Serializable {

    private int correct = 0;
    private int numRowCounter = 0;
    private INDArray nSamples;
    private INDArray rank;
    private double score = 0.0;//or loss
    
    public CatanEvaluation(int maxLabels) {
    	nSamples = Nd4j.zeros(maxLabels);
    	rank = Nd4j.zeros(maxLabels);
    }

    /**
     * Collects statistics on the real outcomes vs the
     * guesses.
     *
     * Note that an IllegalArgumentException is thrown if the two passed in
     * matrices aren't the same length.
     * @param realOutcomes the labels describing which action indices are the correct ones (could be multiple)
     * @param guesses the guesses (usually a probability vector)
     * */
    public void eval(INDArray realOutcomes,INDArray guesses) {
        // Add the number of rows to numRowCounter
        numRowCounter += realOutcomes.shape()[0];
        // and to the nSamples
        nSamples.get(NDArrayIndex.all(),NDArrayIndex.interval(realOutcomes.shape()[1]-1,realOutcomes.shape()[1])).addi(1);
        
        // Length of real labels must be same as length of predicted labels
        if(realOutcomes.length() != guesses.length())
            throw new IllegalArgumentException("Unable to evaluate. Outcome matrices not same length");

        // For each row get the most probable label (column) from prediction and assign as guessMax
        // For each row get the column of the true label and assign as currMax
        for(int i = 0; i < realOutcomes.rows(); i++) {//I think there will only be one row per label
            INDArray currRow = realOutcomes.getRow(i);
            INDArray guessRow = guesses.getRow(i);
            computeScore(currRow, guessRow);
            
            int guessMax;
            HashMap<Integer, Double> guessesMap = new HashMap<Integer, Double>();
            double max = guessRow.getColumn(0).getDouble(i);
            guessMax = 0;
            guessesMap.put(0, max);
            for (int col = 1; col < guessRow.columns(); col++) {
            	guessesMap.put(col,guessRow.getColumn(col).getDouble(i));
                if (guessRow.getColumn(col).getDouble(i) > max) {
                    max = guessRow.getColumn(col).getDouble(i);
                    guessMax = col;
                }
            }
            
            //assumes the correct ones are equal to 1 (could be multiple)
            ArrayList<Integer> correctOutcomes = new ArrayList<Integer>();
            for (int col = 0; col < currRow.columns(); col++) {
            	if(currRow.getColumn(col).getInt(i)==1){
            		correctOutcomes.add(col);
            	}
            }
            
            //for the accuracy, check the best one
            if(correctOutcomes.contains(guessMax)){
            	correct++;
            }
            
            //also create the rank
            guessesMap = sortHashMapByValues(guessesMap); //sort the map based on the values highest first
        	int r = 0;
        	for (Integer key : guessesMap.keySet()) {
        	   if(correctOutcomes.contains(key))
        		   break;//found the correct one, break 
        	   r++;
        	}
        	rank.get(NDArrayIndex.all(),NDArrayIndex.interval(r,r+1)).addi(1);
        }
    }

    // Method to print the classification report
    public String stats() {
        StringBuilder builder = new StringBuilder().append("\n");

        DecimalFormat df = new DecimalFormat("#.####");
        builder.append("\n Accuracy:  " + df.format(accuracy()));
        builder.append("\n Score:  " + df.format(score()));
        builder.append("\n Rank:  " + rank);
        builder.append("\n nSamples:  " + nSamples);
        return builder.toString();
    }

    /**
     * Accuracy:
     * (TP + TN) / (P + N)
     * @return the accuracy of the guesses so far
     */
    public double accuracy() {
        return correct / getNumRowCounter();
    }
    
    public double correct(){
    	return correct;
    }
    
    public double score(){
    	return score / getNumRowCounter();
    }

    public void computeScore(INDArray labels, INDArray output){
    	score += LossCalculation.builder()
                .labels(labels).z(output).lossFunction(LossFunctions.LossFunction.MCXENT)
                .miniBatch(false).miniBatchSize(1)
                .useRegularization(false).build().score();
    }
    
    // Other misc methods
    public void reset(){
    	correct = 0;
    	numRowCounter = 0;
    	nSamples = Nd4j.zeros(nSamples.size(0));
    	rank = Nd4j.zeros(rank.size(0));
    	score = 0.0;
    }
    public double getNumRowCounter() {return (double) numRowCounter;}

    public INDArray getRank(){
    	return rank;
    }
   
    public LinkedHashMap sortHashMapByValues(HashMap passedMap) {
	   List mapKeys = new ArrayList(passedMap.keySet());
	   List mapValues = new ArrayList(passedMap.values());
	   Collections.sort(mapValues);
	   Collections.sort(mapKeys);
	   Collections.reverse(mapValues);
	   Collections.reverse(mapKeys);
	   
	   LinkedHashMap sortedMap = new LinkedHashMap();

	   Iterator valueIt = mapValues.iterator();
	   while (valueIt.hasNext()) {
	       Object val = valueIt.next();
	       Iterator keyIt = mapKeys.iterator();

	       while (keyIt.hasNext()) {
	           Object key = keyIt.next();
	           String comp1 = passedMap.get(key).toString();
	           String comp2 = val.toString();

	           if (comp1.equals(comp2)){
	               passedMap.remove(key);
	               mapKeys.remove(key);
	               sortedMap.put((Integer)key, (Double)val);
	               break;
	           }

	       }

	   }
	   return sortedMap;
	}

}
