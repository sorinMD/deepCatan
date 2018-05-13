package data;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.Iterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import util.DataUtils;
import util.ModelUtils;

/**
 * Normaliser for the Catan data.
 * 
 * @author sorinMD
 *
 */
public class Normaliser {

	private boolean initialised = false;
	private INDArray means = null;
	private INDArray stds = null;
	private String path = ModelUtils.PARAMS_PATH + "normalisation/"; //default, but should be updated via the constructor
	private boolean maskedInput = false;
	
	public Normaliser(String path, boolean maskedInput) {
		this.maskedInput = maskedInput;
		if(maskedInput)
			this.path = path + "/normalisation/maskedInput/";
		else
			this.path = path + "/normalisation/";
	}
	
	/**
	 * For the multi-task case, where multiple tasks are trained at the same
	 * time. Takes in CatanDataSetIterators over the training datasets and
	 * extracts the max, min, mean and standard deviation which can be then used
	 * to normalise each sample as it is received This is required for large
	 * datasets, where keeping everything in memory is not an option
	 * 
	 * @param its
	 */
	public void init(CatanDataSetIterator[] its){
		
        File f = new File(path + "Norm-all.dat");
        if(f.exists()){
        	loadNormalisationParameters(f);
        	return;
        }
    	//make sure the directory structure exists
    	f.getParentFile().mkdirs();
    	
    	
    	if(maskedInput) {
    		//remove task 4 from normalisation since we cannot train on it 
    		CatanDataSetIterator[] its2 = new CatanDataSetIterator[5];	
    		for(int i = 0; i < 4; i++)
    			its2[i] = its[i];
    		its2[4] = its[5];
    		computeZeroMeanUnitVarianceParams(its2, f);
    	}else {
    		computeZeroMeanUnitVarianceParams(its, f);
    	}
	}
	
	/**
	 * Takes in a CatanDataSetIterator over the training dataset and extracts
	 * the max, min, mean and standard deviation which can be then used to
	 * normalise each sample as it is received This is required for large
	 * datasets, where keeping everything in memory is not an option
	 * 
	 * @param it
	 * @param task
	 */
	public void init(CatanDataSetIterator it, int task){
		
        File f = new File(path + "Norm-" + task + ".dat");
        if(f.exists()){
        	loadNormalisationParameters(f);
        	return;
        }
    	//make sure the directory structure exists
    	f.getParentFile().mkdirs();
		
        CatanDataSetIterator[] its = new CatanDataSetIterator[1];
        its[0] = it;
        computeZeroMeanUnitVarianceParams(its, f);
	}
	
	/**
	 * For the 1 to 1 transfer learning case. Takes in a CatanDataSetIterator
	 * over the training dataset and extracts the max, min, mean and standard
	 * deviation which can be then used to normalise each sample as it is
	 * received This is required for large datasets, where keeping everything in
	 * memory is not an option
	 * 
	 * @param its
	 * @param task
	 * @param preTask
	 */
	public void init(CatanDataSetIterator[] its, int task, int preTask){
		
        File f = new File(path + "Norm-" + task + "_" + preTask + ".dat");
        if(f.exists()){
        	loadNormalisationParameters(f);
        	return;
        }else{
        	//check the other combination
        	f = new File(path + "Norm-" + preTask + "_" +  task + ".dat");
            if(f.exists()){
            	loadNormalisationParameters(f);
            	return;
            }
        }
        //make sure the directory structure exists
    	f.getParentFile().mkdirs();
		
        computeZeroMeanUnitVarianceParams(its, f);
	}
	
	private void computeZeroMeanUnitVarianceParams(CatanDataSetIterator[] its, File f){
		//first pass
		long nsamples = 0;
		for(int i = 0; i < its.length ; i++){
			CatanDataSetIterator it = its[i];
			it.reset();
			while (it.hasNext()) {
	         	CatanDataSet cds = it.next();
	         	DataSet ds = DataUtils.turnToSAPairsDS(cds);
	         	Iterator<DataSet> it2 = ds.iterator();
	         	while(it2.hasNext()){
					DataSet ds2 = it2.next();
					//make sure the array is initialised
					if(means == null){
						means = Nd4j.zeros(ds2.getFeatureMatrix().size(1));
					}
					means.addiRowVector(ds2.getFeatureMatrix());
					
					nsamples++;
				}
			}
		}
		//divide to get the mean
		means.divi(nsamples);
		
		//second pass to compute the variance
		for(int i = 0; i < its.length ; i++){
			CatanDataSetIterator it = its[i];
			it.reset();
			while (it.hasNext()) {
	         	CatanDataSet cds = it.next();
	         	DataSet ds = DataUtils.turnToSAPairsDS(cds);
	         	Iterator<DataSet> it2 = ds.iterator();
	         	while(it2.hasNext()){
	         		DataSet ds2 = it2.next();
	         		//make sure the array is initialised
					if(stds == null){
						stds = Nd4j.zeros(ds2.getFeatureMatrix().size(1));
					}
					//compute the variance first
					stds = stds.addiRowVector(Transforms.pow(means.sub(ds2.getFeatureMatrix()),2));
	    		}
	    	}
			it.reset();
		}
		//compute the standard deviation by dividing by the nsamples and squaring
		stds = Transforms.sqrt(stds.divi(nsamples));
		
		//finally write to file
		saveNormalisationParameters(f);
		
		initialised = true;
	}
	
	
	public void normalizeZeroMeanUnitVariance(DataSet ds){
		if(!initialised)
			throw new RuntimeException("Normaliser was not initialised");
		ds.setFeatures(ds.getFeatures().subiRowVector(means));
        ds.setFeatures(ds.getFeatures().diviRowVector(stds.add(Nd4j.scalar(Nd4j.EPS_THRESHOLD))));//avoid NaNs
		
	}
    
    private void saveNormalisationParameters(File f) {   	
        try {
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f));
            DataOutputStream dis = new DataOutputStream(bos);
            Nd4j.write(means,dis);
            Nd4j.write(stds,dis);
            dis.flush();
            dis.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public void loadNormalisationParameters(File f) {
        try {
            BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f));
            DataInputStream dis = new DataInputStream(bis);
        	means = Nd4j.read(dis);
        	stds = Nd4j.read(dis);
        	dis.close();
        	initialised = true;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
	
}
