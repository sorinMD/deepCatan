package org.deeplearning4j.nn.multilayer;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.layers.CustomOutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Same as the {@link MultiLayerNetwork} class, but setting the number of actions for each sample before training/evaluating
 * 
 * @author sorinMD
 *
 */
public class CatanMlp extends MultiLayerNetwork {
    
	private MultiLayerConfiguration fullConfiguration;
	
    public MultiLayerConfiguration getFullConfig(){
    	return fullConfiguration;
    }
	
    /**
     * the number of times we need to replicate the final layers as a column vector
     */
    protected INDArray numberOfActions;
    
    public void setNumberOfActions(INDArray nAct){
    	((CustomOutputLayer)getOutputLayer()).setNumberOfActions(nAct);
    	numberOfActions = nAct;
    }
    
	public CatanMlp(MultiLayerConfiguration conf) {
		super(conf);
		fullConfiguration = conf;
	}

    /**
     * Fit the model; NOTE: only via backprop and without normalising data;
     *
     * @param data the data to train on
     */
    public void fit(DataSet data, INDArray nActions) {
    	//sets the action input and the size of the set of legal actions
    	setNumberOfActions(nActions);
        fit(data.getFeatureMatrix(), data.getLabels());
    }
    
    /**
     * Output on a dataset (it can be made of a single sample)
     * Note: Not for training
     *
     * @param data the data to test on
     */
    public INDArray output(DataSet data, INDArray nActions) {
    	//sets the action input and the size of the set of legal actions
    	setNumberOfActions(nActions);
    	//remember to set the action input before the state input
        return output(data.getFeatureMatrix().dup(),false);
    }
            
}
