package data;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

/**
 * Data format for storing the state features, action features for all legal
 * actions, the target action (i.e. the features of the target action) and the
 * number of legal actions.
 * 
 * @author sorinMD
 *
 */
public class CatanDataSet {
	
    private INDArray stateFeatures;
    /**
     * The set of all legal actions from this state, listed as features
     */
    private INDArray actionFeatures;
    /**
     * The array containing the features of the chosen action;
     */
    private INDArray targetAction;
    //other specific fields that may help during training;
	/**
	 * The {@link #actionFeatures} contain all legal actions, this field tells
	 * us how many sets of action features are included
	 */
    private INDArray actionSetSize;
    
    public CatanDataSet() {
        this(Nd4j.zeros(new int[]{1,1}), Nd4j.zeros(new int[]{1,1}), Nd4j.zeros(new int[]{1,1}), Nd4j.zeros(new int[]{1,1}));
    }

    /**
     * Creates a dataset with the specified input matrix and labels
     *
     * @param first  the state feature matrix
     * @param second the actions feature matrix
     * @param third the target action
     * @param fourth the size of the set of legal actions for each sample (as a column vector)
      */
    public CatanDataSet(INDArray first, INDArray second, INDArray third, INDArray fourth) {
        if (first.size(0) != third.size(0) || first.size(0) != fourth.size(0))
            throw new IllegalStateException("Invalid data transform; arrays do not have equal rows. First was " + first.size(0) + " third was " + third.size(0) + " and fourth was " + fourth.size(0));
        this.stateFeatures = first;
        this.actionFeatures = second;
        this.targetAction = third;
        this.actionSetSize = fourth;
    }
	
    public int numExamples() {
        return getStateFeatures().size(0);
    }
    
    public INDArray getStateFeatures() {
        return stateFeatures;
    }
    
    public INDArray getActionFeatures() {
        return actionFeatures;
    }
    
    public INDArray getTargetAction() {
        return targetAction;
    }
    
    public INDArray getSizeOfLegalActionSet(){
    	return actionSetSize;
    }
    
    public void setStateFeatures(INDArray features) {
        this.stateFeatures = features;
    }
    
    public void setActionFeatures(INDArray features) {
        this.actionFeatures = features;
    }
    
    public void setTargetAction(INDArray labels) {
        this.targetAction = labels;
    }
    
    public void load(File from) {
        try {
            BufferedInputStream bis = new BufferedInputStream(new FileInputStream(from));
            DataInputStream dis = new DataInputStream(bis);
            stateFeatures = Nd4j.read(dis);
            actionFeatures = Nd4j.read(dis);
            targetAction = Nd4j.read(dis);
            actionSetSize = Nd4j.read(dis);
            dis.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void save(File to) {
        try {
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(to));
            DataOutputStream dis = new DataOutputStream(bos);
            Nd4j.write(getStateFeatures(),dis);
            Nd4j.write(getActionFeatures(),dis);
            Nd4j.write(getTargetAction(),dis);
            Nd4j.write(getSizeOfLegalActionSet(),dis);
            dis.flush();
            dis.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    /**
     * Clone the dataset
     *
     * @return a clone of the dataset
     */
    public CatanDataSet copy() {
        CatanDataSet ret = new CatanDataSet(getStateFeatures().dup(), getActionFeatures().dup(), getTargetAction().dup(), actionSetSize.dup());
        return ret;
    }
    
    /**
     * Only applicable to the state input
     * @param num
     */
    public void multiplyBy(double num) {
        getStateFeatures().muli(Nd4j.scalar(num));
    }

    /**
     * Only applicable to the state input
     * @param num
     */
    public void divideBy(int num) {
        getStateFeatures().divi(Nd4j.scalar(num));
    }
    
    /**
     * Only applicable to the state input
     * @param min
     * @param max
     */
    public void scaleMinAndMax(double min, double max) {
        FeatureUtil.scaleMinMax(min, max, getStateFeatures());
    }
    
    /**
     * Based on the number of states so equal to the number of samples
     * @return
     */
    public int numInputs() {
        return getStateFeatures().columns();
    }
    
    /**
     * Number of inputs based on the number of actions
     * @return
     */
    public int numActionInputs(int index) {
        return actionSetSize.getRow(index).getInt(0);//there is only one per row
    }
    
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("===========STATE INPUT===================\n")
                .append(getStateFeatures().toString().replaceAll(";", "\n"))
                .append("===========ACTION INPUT===================\n")
                .append(getActionFeatures().toString().replaceAll(";", "\n"))
                .append("\n=================OUTPUT==================\n")
                .append(getTargetAction().toString().replaceAll(";", "\n"));
        return builder.toString();
    }
    
    /**
     * Not recommended as it may result in different actions or states being the same
     * Same as calling binarizeInput(0)
     */
    public void binarizeInput() {
        binarizeInput(0);
    }

    /**
     * Not recommended as it may result in different actions or states being the same
     * Binarizes the dataset such that any number greater than cutoff is 1 otherwise zero
     * @param cutoff the cutoff point
     */
    public void binarizeInput(double cutoff) {
        INDArray linear = getStateFeatures().linearView();
        for (int i = 0; i < getStateFeatures().length(); i++) {
            double curr = linear.getDouble(i);
            if (curr > cutoff)
                getStateFeatures().putScalar(i, 1);
            else
                getStateFeatures().putScalar(i, 0);
        }
        
        linear = getActionFeatures().linearView();
        for (int i = 0; i < getActionFeatures().length(); i++) {
            double curr = linear.getDouble(i);
            if (curr > cutoff)
                getActionFeatures().putScalar(i, 1);
            else
                getActionFeatures().putScalar(i, 0);
        }
    }
    
    /**
     * Not recommended as it may result in different actions being the same and so multiple input options being equal to the action label
     * Same as calling binarizeOutput(0)
     */
    public void binarizeOutput() {
        binarizeInput(0);
    }

    /**
     * Not recommended as it may result in different actions being the same and so multiple input options being equal to the action label
     * Binarizes the dataset such that any number greater than cutoff is 1 otherwise zero
     * @param cutoff the cutoff point
     */
    public void binarizeOutput(double cutoff) {
        INDArray linear = getTargetAction().linearView();
        for (int i = 0; i < getTargetAction().length(); i++) {
            double curr = linear.getDouble(i);
            if (curr > cutoff)
            	getTargetAction().putScalar(i, 1);
            else
            	getTargetAction().putScalar(i, 0);
        }
    }
    
    /**
     * Squeezes input data to a max and a min
     *
     * @param min the min value to occur in the dataset
     * @param max the max value to ccur in the dataset
     */
    public void squishToRange(double min, double max) {
        for (int i = 0; i < getStateFeatures().length(); i++) {
            double curr = (double) getStateFeatures().getScalar(i).element();
            if (curr < min)
                getStateFeatures().put(i, Nd4j.scalar(min));
            else if (curr > max)
                getStateFeatures().put(i, Nd4j.scalar(max));
        }
        for (int i = 0; i < getActionFeatures().length(); i++) {
            double curr = (double) getActionFeatures().getScalar(i).element();
            if (curr < min)
                getActionFeatures().put(i, Nd4j.scalar(min));
            else if (curr > max)
            	getActionFeatures().put(i, Nd4j.scalar(max));
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof CatanDataSet)) return false;

        CatanDataSet dataSet = (CatanDataSet) o;

        if (getStateFeatures() != null ? !getStateFeatures().equals(dataSet.getStateFeatures()) : dataSet.getStateFeatures() != null)
            return false;
        if (getActionFeatures() != null ? !getActionFeatures().equals(dataSet.getActionFeatures()) : dataSet.getActionFeatures() != null)
            return false;
        return !(getTargetAction() != null ? !getTargetAction().equals(dataSet.getTargetAction()) : dataSet.getTargetAction() != null);

    }

    @Override
    public int hashCode() {
        int result = getStateFeatures() != null ? getStateFeatures().hashCode() : 0;
        result = 31 * result + (getActionFeatures() != null ? getActionFeatures().hashCode() : 0);
        result = 31 * result + (getTargetAction() != null ? getTargetAction().hashCode() : 0);
        return result;
    }
    
    /**
     * @return an equivalent of this dataset as the standard datastructure, without the vector containing the number of legal actions
     */
    public DataSet getDataSet(){
    	return new DataSet(Nd4j.hstack(this.getStateFeatures(),this.getActionFeatures()), this.getTargetAction());
    }
    
    
}
