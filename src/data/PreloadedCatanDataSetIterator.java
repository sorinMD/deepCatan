package data;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

/**
 * Iterator that reads all the data beforehand and keeps it in memory.
 * 
 * @author sorinMD
 *
 */
public class PreloadedCatanDataSetIterator{

	ArrayList<CatanDataSet> data;
	int counter = 0;
	int batchSize = 1;
	
	public PreloadedCatanDataSetIterator(ArrayList<CatanDataSet> data, int batchSize) {
		this.data = data;
		counter = 0;
		this.batchSize = batchSize;
	}
	
	public boolean hasNext() {
		if(data.size() - counter >= batchSize)
			return true;
		return false;
	}

	public CatanDataSet next() {
		return next(batchSize);
	}

	public CatanDataSet next(int num) {
		List<CatanDataSet> dataSets = new ArrayList<>();
		for (int i = 0; i < num; i++) {
			if (!hasNext())
				break;
			dataSets.add(data.get(counter));
			counter++;
		}

		if (dataSets.size() == 1)
			return dataSets.get(0);
		
		//the following code is for minibatching
		int actRows = 0;
		int actColumns = dataSets.get(0).getTargetAction().columns();
		int stateColumns = dataSets.get(0).getStateFeatures().columns();
		for (CatanDataSet data : dataSets) {
			actRows += data.getActionFeatures().rows();
		}
		// NOTE: this is awful code, but vstack doesn't work well with minibatches in this dl4j version
		// for some obscure reason
		INDArray stateInput = Nd4j.create(dataSets.size(), stateColumns);
		INDArray actionInput = Nd4j.create(actRows, actColumns);
		INDArray label = Nd4j.create(dataSets.size(), actColumns);
		INDArray actionSetSize = Nd4j.create(dataSets.size(), 1);
		int actIdx = 0;
		for (int i = 0; i < dataSets.size(); i++) {
			CatanDataSet data = dataSets.get(i);
			stateInput.putRow(i, data.getStateFeatures());
			label.putRow(i, data.getTargetAction());
			actionSetSize.putRow(i, data.getSizeOfLegalActionSet());
			for (int j = 0; j < data.getActionFeatures().rows(); j++) {
				actionInput.putRow(actIdx, data.getActionFeatures().getRow(j));
				actIdx++;
			}
		}

		CatanDataSet ret = new CatanDataSet(stateInput, actionInput, label, actionSetSize);
		return ret;
	}
	
	public int totalExamples() {
		return data.size();
	}

	public void reset() {
		counter = 0;
	}

	public int batch() {
		return batchSize;
	}

	public int currentCounter() {
		return counter;
	}
}
