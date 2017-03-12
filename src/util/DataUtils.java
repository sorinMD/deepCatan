package util;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import data.CatanDataSet;

public class DataUtils {
	
	/**
	 * Turns the CatanDataSet into a DataSet (of state-action and label) based on both the cosine similarity and the real label
	 * This is equivalent to softening of the labels either with additional information on the problem (cosine similarity) or
	 * just by reducing the weight attributed to the correct class and distributing it across the other classes. It has a strong
	 * regularization effect.
	 * @param cds
	 * @param metricW
	 * @param labelW
	 * @return
	 */
    public static DataSet turnToSAPairsDS(CatanDataSet cds, double metricW, double labelW){
    	int nSamples = cds.numExamples();
    	List<INDArray> inputs = new ArrayList<>();//both state and action
        List<INDArray> labels = new ArrayList<>();
    	int actIndex = 0;//we have multiple rows per sample here, so keep track of this index with a counter
    	Accumulation acc;
    	
    	for(int j = 0; j < nSamples; j++){
    		INDArray labelAction;
    		double[] cosArray = new double[cds.numActionInputs(j)];
    		double[] yArray = new double[cds.numActionInputs(j)];
    		if(nSamples == 1){
    			labelAction = cds.getTargetAction();
    		}else{
    			labelAction = cds.getTargetAction().getRow(j);
    		}
			
	    	for(int i = 0; i < cds.numActionInputs(j); i++){
	    		acc = new CosineSimilarity(cds.getActionFeatures().getRow(actIndex), labelAction);
	    		Nd4j.getExecutioner().exec(acc);
	    		
    	    	if(Double.isNaN(acc.currentResult().doubleValue())){
	    			/*in the case of the robber, there are actions that do not modify the state description and the cosine distance could be NaN
	    			 	We handle this via an extreme method of checking if it is the same action (1) or not (-1) */
    	    		if(cds.getActionFeatures().getRow(actIndex).equals(labelAction)){
    	    			cosArray[i] = 1;
    	    		}else{
    	    			cosArray[i] = -1;
    	    		}
	    		}else
	    			cosArray[i] = acc.currentResult().doubleValue();
    	    	
	    		//if it is the same action then set the y to 1, otherwise 0
    	    	if(cds.getActionFeatures().getRow(actIndex).equals(labelAction)){
    	    		yArray[i] = 1;
    	    	}
    	    	inputs.add(Nd4j.hstack(cds.getStateFeatures().getRow(j),cds.getActionFeatures().getRow(actIndex)));
	    		actIndex++;
	    	}
	    	INDArray cos = Nd4j.create(cosArray);
	    	INDArray y = Nd4j.create(yArray);
	    	INDArray weighted = cos.mul(metricW).add(y.mul(labelW));
	    	SoftMax softMax = new SoftMax(weighted);
	    	softMax.exec(1);
	    	labels.add(softMax.z().transpose());
    	}
    	return new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])), Nd4j.vstack(labels.toArray(new INDArray[0])));
    }
    
    /**
     * Turns the action features into labels based on the action inputs per data sample !!!!
     * Pre-process step required for evaluation, as the CatanDataSet contains the correct action as a vector of features
     */
    public static INDArray computeLabels(CatanDataSet data) {
    	if(data.numExamples() > 1)
    		throw new RuntimeException("Cannot compute labels for minibatch");
    	double[] array = new double[data.getSizeOfLegalActionSet().getInt(0)];
    	for(int i = 0; i < data.getSizeOfLegalActionSet().getInt(0); i++){
    		//if it is the same action then set the y to 1
    		if(data.getActionFeatures().getRow(i).equals(data.getTargetAction())){
    			array[i] = 1;
    		}
    	}
    	
    	return Nd4j.create(array);
    }
    
    
    /**
     * Turns the CatanDataSet into a DataSet with the real labels only
     * @param cds
     * @return
     */
    public static DataSet turnToSAPairsDS(CatanDataSet catands){
    	int nSamples = catands.numExamples();
    	List<INDArray> inputs = new ArrayList<>();//both state and action
        List<INDArray> labels = new ArrayList<>();
    	int actIndex = 0;//we have multiple rows per sample here, so keep track of this index with a counter
    	
    	for(int j = 0; j < nSamples; j++){
    		INDArray labelAction;
    		double[] array = new double[catands.numActionInputs(j)];
    		if(nSamples == 1){
    			labelAction = catands.getTargetAction();
    		}else{
    			labelAction = catands.getTargetAction().getRow(j);
    		}
	    	for(int i = 0; i < catands.numActionInputs(j); i++){
	    		//if it is the same action then set the y to 1
	    		if(catands.getActionFeatures().getRow(actIndex).equals(labelAction)){
	    			array[i] = 1;
	    		}
	    		inputs.add(Nd4j.hstack(catands.getStateFeatures(),catands.getActionFeatures().getRow(actIndex)));
	    		actIndex++;
	    	}
	    	labels.add(Nd4j.create(array).transposei());
    	}
    	
	    return new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])), Nd4j.vstack(labels.toArray(new INDArray[0])));
    }
    	
}
