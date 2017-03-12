package data;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.arbiter.data.DataSetIteratorProvider;
import org.deeplearning4j.datasets.creator.DataSetIteratorFactory;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Simple data iterator that reads the data from a given file and turns it into
 * a CatanDataSet data structure.
 * 
 * @author sorinMD
 */
public class CatanDataSetIterator {
	//fields required for control
	private File file;
	private RecordReader recordReader;
    private boolean overshot = false;
    private CatanDataSet last;
	//fields specific to our data format
    private int batchSize = 1;
    private int stateIndex = 1;
    private int actionLabelIndex = -1;
    private int legalActionsIndex = -1;
    private int index = 0;
    private int nSamples;
    private boolean stackedActions = true;
       
   /**
    * 
    * @param recordReader
    * @param converter
    * @param batchSize
    * @param actionlabelIndex
    * @param legalActionsIndex
    */
   public CatanDataSetIterator(File f, int nSamples, int batchSize, int actionlabelIndex, int legalActionsIndex, boolean sa) {
       file = f;
	   recordReader = new CSVRecordReader();
       try {
			recordReader.initialize(new FileSplit(f));
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
       this.batchSize = batchSize;
       this.actionLabelIndex = actionlabelIndex;
       this.legalActionsIndex = legalActionsIndex;
       this.nSamples = nSamples;
       this.stackedActions = sa;
   }
   
   /**
    * 
    * @param recordReader
    * @param converter
    * @param batchSize
    * @param actionlabelIndex
    * @param legalActionsIndex
    */
   public CatanDataSetIterator(RecordReader recordReader, int nSamples, int batchSize, int actionlabelIndex, int legalActionsIndex, boolean sa) {
       this.recordReader = recordReader;
       this.batchSize = batchSize;
       this.actionLabelIndex = actionlabelIndex;
       this.legalActionsIndex = legalActionsIndex;
       this.nSamples = nSamples;
       this.stackedActions = sa;
   }
   
   public CatanDataSet next(int num) {

       List<CatanDataSet> dataSets = new ArrayList<>();
       for (int i = 0; i < num; i++) {
           if (!hasNext())
               break;
           Collection<Writable> record = recordReader.next();
           CatanDataSet cds = getDataSet(record);
           //ignore cases where there is a single option, in case these have been missed before when reading from hte database
           if(cds.getActionFeatures().shape()[0] > 1){
        	   dataSets.add(cds);
           }else{
        	   i--;//repeat this step
           }
           index++;
       }
       
       if(dataSets.isEmpty()) {
           overshot = true;
           return last;
       }
       
       if(dataSets.size()==1){
    	  return dataSets.get(0);
       }
       
       //there is no better way of computing the action rows
       int actRows = 0;
       int actColumns = dataSets.get(0).getTargetAction().columns();
       int stateColumns = dataSets.get(0).getStateFeatures().columns();
       for(CatanDataSet data : dataSets){
    	   actRows+=data.getActionFeatures().rows();
       }
       //NOTE: this is awful code, but vstack doesn't work with matrices for some obscure reason
       INDArray stateInput = Nd4j.create(dataSets.size(), stateColumns);
       INDArray actionInput = Nd4j.create(actRows, actColumns);
       INDArray label = Nd4j.create(dataSets.size(), actColumns);
       INDArray actionSetSize = Nd4j.create(dataSets.size(), 1);
       int actIdx = 0;
       for ( int i= 0; i < dataSets.size(); i++) {
    	   CatanDataSet data = dataSets.get(i);
    	   stateInput.putRow(i, data.getStateFeatures());
    	   label.putRow(i, data.getTargetAction());
    	   actionSetSize.putRow(i, data.getSizeOfLegalActionSet());
    	   for(int j = 0; j < data.getActionFeatures().rows(); j++){
    		   actionInput.putRow(actIdx,data.getActionFeatures().getRow(j));
    		   actIdx++;
    	   }
       }
       
       CatanDataSet ret = new CatanDataSet(stateInput, actionInput, label, actionSetSize);
       last = ret;
       return ret;
   }

   private CatanDataSet getDataSet(Collection<Writable> record) {
       List<Writable> currList;
       if (record instanceof List)
           currList = (List<Writable>) record;
       else
           currList = new ArrayList<>(record);
       
       //do not read the biases for any of them
       INDArray setLegalActionsSize = Nd4j.create(1).assign(Double.valueOf(currList.get(0).toString()));
       INDArray stateFeatures = Nd4j.create(actionLabelIndex - stateIndex - 1);
       INDArray actionLabel = Nd4j.create(legalActionsIndex - actionLabelIndex - 1);
       INDArray af = Nd4j.create((int) Double.parseDouble(currList.get(0).toString()), actionLabel.size(1));
       INDArray actionFeatures = Nd4j.create(currList.size() - legalActionsIndex - (int) Double.parseDouble(currList.get(0).toString()));
       
       int idx = 2; //ignore the size of the action set and the state bias
       //fill the state
       for(int i=0; i<stateFeatures.length(); i++){
    	   stateFeatures.putScalar(i, Double.valueOf(currList.get(idx).toString()));
    	   idx++;
       }
       idx++;//get rid of the action label bias
       //fill the actionLabel
       for(int i=0; i<actionLabel.length(); i++){
    	   actionLabel.putScalar(i, Double.valueOf(currList.get(idx).toString()));
    	   idx++;
       }
       
       //fill the possible actions
       int rIdx = -1;//the row index //it is incremented once before starting the loop below
       int cIdx = 0;//the column index
       for(int i=0; i<actionFeatures.length(); i++){
    	   if((double) i % actionLabel.size(1) == 0){
    		   idx++;//do not add any of the biases
    		   rIdx++;
    		   cIdx = 0;
    	   }
    	   actionFeatures.putScalar(i, Double.valueOf(currList.get(idx).toString()));
    	   af.putScalar(rIdx, cIdx, Double.valueOf(currList.get(idx).toString()));
    	   idx++;
    	   cIdx++;
       }
       
       if(stackedActions)     
    	   return new CatanDataSet(stateFeatures, af, actionLabel, setLegalActionsSize);
       else
    	   return new CatanDataSet(stateFeatures, actionFeatures, actionLabel, setLegalActionsSize);
   }
   
   public void reset(){
		try {
		    recordReader.close();
		    recordReader = new CSVRecordReader();
			recordReader.initialize(new FileSplit(file));
			index = 0;
			overshot = false;
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
   }
   
   public boolean hasNext() {
       return (recordReader.hasNext() || overshot) && index < nSamples;
   }
   
   public CatanDataSet next() {
       return next(batchSize);
   }
   
   public int batch(){
	   return batchSize;
   }
   
   /**
    * Collapses the two inputs (state and action features) into a single input and returns a DataSet
    * @return
    */
   public DataSet nextDataSet() {
	   CatanDataSet cds;
	   if(!stackedActions)
         cds = next(batchSize);
	   else 
		   throw new IllegalStateException("Can't turn next batch into DataSet due to stackedActions"); //perhaps not the correct exception class...
      
	   return new DataSet(Nd4j.hstack(cds.getStateFeatures(),cds.getActionFeatures()), cds.getTargetAction());
   }
   
}
