package util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Simple static utility that masks the features that should not be observable by the opponents in a normal Catan game.
 * Even though this will make the training slow, it is needed to avoid processing the data again and storing it twice.
 * 
 * @author sorinMD
 *
 */
public class CatanFeatureMaskingUtil implements NumericalFeatureVectorOffsets {
	
	//TODO: update the count of dropped features if the masking procedure changes
	public static int droppedFeaturesCount = 17; 
	
	public static INDArray maskState(INDArray state) {
		//the -1 is needed to account for the bias which dl4j keeps it separate
		INDArray first = state.get(NDArrayIndex.all(), NDArrayIndex.interval(0,OFS_CLAYINHAND-1)).dup();
		INDArray rssSum = state.get(NDArrayIndex.all(), NDArrayIndex.interval(OFS_CLAYINHAND-1,OFS_WOODINHAND)).sum(1);
		INDArray second = state.get(NDArrayIndex.all(), NDArrayIndex.interval(OFS_WOODINHAND,OFS_OLDDEVCARDSINHAND-1)).dup();
		INDArray devSum = state.get(NDArrayIndex.all(), NDArrayIndex.interval(OFS_OLDDEVCARDSINHAND-1,OFS_VPCARDS)).sum(1); 
		INDArray third = state.get(NDArrayIndex.all(), NDArrayIndex.interval(OFS_OVER7CARDS-1,state.size(1))).dup();
		//update the score based on the vp cards since the opponents cannot know the real score if they cannot see the vp cards
		first.getColumn(OFS_PLAYERSCORE-1).subi(state.getColumn(OFS_VPCARDS-1));
		//form a new vector with the remaining data
		return Nd4j.hstack(first,rssSum,second,devSum,third);
	}
	
	
	public static INDArray maskAction(INDArray action) {
		//the -1 is needed to account for the bias which dl4j keeps it separate
		INDArray first = action.get(NDArrayIndex.all(), NDArrayIndex.interval(0,OFS_ACT_RSSINHAND-1)).dup();
		INDArray rssSum = action.get(NDArrayIndex.all(), NDArrayIndex.interval(OFS_ACT_RSSINHAND-1,OFS_ACT_RSSINHAND+4)).sum(1);
		INDArray second = action.get(NDArrayIndex.all(), NDArrayIndex.interval(OFS_ACT_RSSINHAND+4,OFS_ACT_OLDDEVCARDSINHAND-1)).dup();
		INDArray devSum = action.get(NDArrayIndex.all(), NDArrayIndex.interval(OFS_ACT_OLDDEVCARDSINHAND-1,OFS_ACT_VPCARDS)).sum(1); 
		INDArray third = action.get(NDArrayIndex.all(), NDArrayIndex.interval(OFS_ACT_NPLAYEDKNIGHTS-1, OFS_ACT_CANBUYCARD-1)).dup();
		INDArray fourth = action.get(NDArrayIndex.all(), NDArrayIndex.interval(OFS_ACT_OVER7CARDS-1,action.size(1))).dup();
		//Note: unfortunately there was a bug in the data collection process and the buying of a development card did not affect the score in the observable games(even though buying a vp card could)
		//therefore there is no need to mask it and the below code is commented out;
		//first.getColumn(OFS_ACT_PLAYERSCORE-1).subi(action.getColumn(OFS_ACT_VPCARDS-1));
		//form a new vector with the remaining data
		return Nd4j.hstack(first,rssSum,second,devSum,third,fourth);
	}
	
	/**
	 * This is used during testing in the StacSettlers framework.
	 * @param stateAction
	 * @return
	 */
	public static INDArray maskConcatenatedArrays(INDArray stateAction) {
		INDArray state = stateAction.get(NDArrayIndex.all(), NDArrayIndex.interval(0,STATE_VECTOR_SIZE-2));
		INDArray action = stateAction.get(NDArrayIndex.all(), NDArrayIndex.interval(STATE_VECTOR_SIZE-2,stateAction.size(1)));
		state = maskState(state);
		action = maskAction(action);
		return Nd4j.hstack(state,action);
	}
}
