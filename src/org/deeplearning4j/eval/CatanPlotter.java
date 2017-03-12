package org.deeplearning4j.eval;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

//remnants from version 34 of dl4j
//import org.deeplearning4j.plot.NeuralNetPlotter;

/**
 * A simple class for plotting and storing all the data from the evaluation of the model
 * NOTE: it only stores all the data into files and the plotting needs to be done in matlab or python;
 * @author sorinMD
 * TODO: sort this thing out as it seems the plotter class was removed in the later versions.
 * TODO: get rid of all the silly code duplication in the write methods
 */
public class CatanPlotter {
	private ArrayList<Double> evalScore = new ArrayList<>();
	private ArrayList<Double> trainScore = new ArrayList<>();
	private ArrayList<Double> evalAccuracy = new ArrayList<>();
	private ArrayList<Double> trainAccuracy = new ArrayList<>();
	private ArrayList<INDArray> evalRank = new ArrayList<>();
	private ArrayList<INDArray> trainRank = new ArrayList<>();
	private ArrayList<INDArray> evalFeatureConf = new ArrayList<>();
	private ArrayList<INDArray> trainFeatureConf = new ArrayList<>();
//	private NeuralNetPlotter eSPlotter = new NeuralNetPlotter();
//	private NeuralNetPlotter tSPlotter = new NeuralNetPlotter();
//	private NeuralNetPlotter eAPlotter = new NeuralNetPlotter();
//	private NeuralNetPlotter tAPlotter = new NeuralNetPlotter();
	public static final String RESULTS_DIR = "results/";
	private int task = 0;
	
	public CatanPlotter(int t){
		//don't need to initialise anything yet, just make sure the directory exists
		task = t;
        File dir = new File(RESULTS_DIR);
    	if(!dir.exists())
        	dir.mkdirs();
//    	eSPlotter.getLayerGraphFilePath();
	}
	
	public void addScores(double es, double ts){
		evalScore.add(es);
		trainScore.add(ts);
	}
	
	public void addAccuracy(double ea, double ta){
		evalAccuracy.add(ea);
		trainAccuracy.add(ta);
	}
	
	public void addRanks(INDArray tr, INDArray er){
		trainRank.add(tr);
		evalRank.add(er);
	}
	
	public void addData(double es, double ts, double ea, double ta){
		evalScore.add(es);
		trainScore.add(ts);
		evalAccuracy.add(ea);
		trainAccuracy.add(ta);
	}
	
	public void addFeatureConfusion(double[] eConf, double[] tConf){
		evalFeatureConf.add(Nd4j.create(eConf));
		trainFeatureConf.add(Nd4j.create(tConf));
	}
	
//	public void plotAll(){
//		String dataFilePath = eAPlotter.writeArray(evalAccuracy);
//		eAPlotter.renderGraph("accuracy", dataFilePath, RESULTS_DIR + "evalAccuracy-" + task + ".png");
//		dataFilePath = tAPlotter.writeArray(trainAccuracy);
//		tAPlotter.renderGraph("accuracy", dataFilePath, RESULTS_DIR + "trainAccuracy-" + task + ".png");
//		dataFilePath = eSPlotter.writeArray(evalAccuracy);
//		eSPlotter.renderGraph("loss", dataFilePath, RESULTS_DIR + "evalScore-" + task + ".png");
//		dataFilePath = tSPlotter.writeArray(trainAccuracy);
//		tSPlotter.renderGraph("loss", dataFilePath, RESULTS_DIR + "trainScore-" + task + ".png");
//	}
	
	/**
	 * Writes everything, but excluding the standard deviations. If the deviations need to be included, call each write separately.
	 */
	public void writeAll(){
		writeEvalAcc(null,null);
		writeTrainAcc(null,null);
		writeEvalScore(null,null);
		writeTrainScore(null,null);
		writeRanks(null,null);
		writeFeatureConfusions(null,null);
	}
	
	public void writeEvalAcc(double[] stdDev, double[] stdErr){
		try {
            File write = new File(RESULTS_DIR + "EvalAccuracy-" + task + ".txt");
			write.delete();
			write.createNewFile();
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
            StringBuilder sb = new StringBuilder();
            for(Object value : evalAccuracy) {
                sb.append(String.format("%.10f", (Double) value));
                sb.append(",");
            }
            String line = sb.toString();
            line = line.substring(0, line.length()-1);
            bos.write(line.getBytes());
            if(stdDev!=null){
            	sb = new StringBuilder();
            	bos.write("\n".getBytes());
                for(double value : stdDev) {
                    sb.append(String.format("%.10f", value));
                    sb.append(",");
                }
                line = sb.toString();
                line = line.substring(0, line.length()-1);
                bos.write(line.getBytes());
            }
            if(stdErr!=null){
            	sb = new StringBuilder();
            	bos.write("\n".getBytes());
                for(double value : stdErr) {
                    sb.append(String.format("%.10f", value));
                    sb.append(",");
                }
                line = sb.toString();
                line = line.substring(0, line.length()-1);
                bos.write(line.getBytes());
            }
            bos.flush();
            bos.close();

        } catch(IOException e){
            throw new RuntimeException(e);
        }
	}
	
	public void writeTrainAcc(double[] stdDev, double[] stdErr){
		try {
			File write = new File(RESULTS_DIR + "TrainAccuracy-" + task + ".txt");
			write.delete();
			write.createNewFile();
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
            StringBuilder sb = new StringBuilder();
            for(Object value : trainAccuracy) {
                sb.append(String.format("%.10f", (Double) value));
                sb.append(",");
            }
            String line = sb.toString();
            line = line.substring(0, line.length()-1);
            bos.write(line.getBytes());
            if(stdDev!=null){
            	sb = new StringBuilder();
            	bos.write("\n".getBytes());
                for(double value : stdDev) {
                    sb.append(String.format("%.10f", value));
                    sb.append(",");
                }
                line = sb.toString();
                line = line.substring(0, line.length()-1);
                bos.write(line.getBytes());
            }
            if(stdErr!=null){
            	sb = new StringBuilder();
            	bos.write("\n".getBytes());
                for(double value : stdErr) {
                    sb.append(String.format("%.10f", value));
                    sb.append(",");
                }
                line = sb.toString();
                line = line.substring(0, line.length()-1);
                bos.write(line.getBytes());
            }
            bos.flush();
            bos.close();

        } catch(IOException e){
            throw new RuntimeException(e);
        }
	}
	
	public void writeTrainScore(double[] stdDev, double[] stdErr){
		try {
            File write = new File(RESULTS_DIR + "TrainScore-" + task + ".txt");
			write.delete();
			write.createNewFile();
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
            StringBuilder sb = new StringBuilder();
            for(Object value : trainScore) {
                sb.append(String.format("%.10f", (Double) value));
                sb.append(",");
            }
            String line = sb.toString();
            line = line.substring(0, line.length()-1);
            bos.write(line.getBytes());
            if(stdDev!=null){
            	sb = new StringBuilder();
            	bos.write("\n".getBytes());
                for(double value : stdDev) {
                    sb.append(String.format("%.10f", value));
                    sb.append(",");
                }
                line = sb.toString();
                line = line.substring(0, line.length()-1);
                bos.write(line.getBytes());
            }
            if(stdErr!=null){
            	sb = new StringBuilder();
            	bos.write("\n".getBytes());
                for(double value : stdErr) {
                    sb.append(String.format("%.10f", value));
                    sb.append(",");
                }
                line = sb.toString();
                line = line.substring(0, line.length()-1);
                bos.write(line.getBytes());
            }
            bos.flush();
            bos.close();

        } catch(IOException e){
            throw new RuntimeException(e);
        }
	}
	
	public void writeEvalScore(double[] stdDev, double[] stdErr){
		try {
            File write = new File(RESULTS_DIR + "EvalScore-" + task + ".txt");
			write.delete();
			write.createNewFile();
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
            StringBuilder sb = new StringBuilder();
            for(Object value : evalScore) {
                sb.append(String.format("%.10f", (Double) value));
                sb.append(",");
            }
            String line = sb.toString();
            line = line.substring(0, line.length()-1);
            bos.write(line.getBytes());
            if(stdDev!=null){
            	sb = new StringBuilder();
            	bos.write("\n".getBytes());
                for(double value : stdDev) {
                    sb.append(String.format("%.10f", value));
                    sb.append(",");
                }
                line = sb.toString();
                line = line.substring(0, line.length()-1);
                bos.write(line.getBytes());
            }
            if(stdErr!=null){
            	sb = new StringBuilder();
            	bos.write("\n".getBytes());
                for(double value : stdErr) {
                    sb.append(String.format("%.10f", value));
                    sb.append(",");
                }
                line = sb.toString();
                line = line.substring(0, line.length()-1);
                bos.write(line.getBytes());
            }
            bos.flush();
            bos.close();

        } catch(IOException e){
            throw new RuntimeException(e);
        }
	}
	
	public void writeRanks(ArrayList<INDArray> stdEval, ArrayList<INDArray> stdTrain){
		try {
            File write = new File(RESULTS_DIR + "Ranks-" + task + ".txt");
			write.delete();
			write.createNewFile();
			FileWriter fileWriter = new FileWriter(write);
			
			for(int i = 0; i < trainRank.size(); i++){
				fileWriter.append("Iteration: " + i);
				fileWriter.append("\n");
		        fileWriter.append("Evaluation rank: ");
		        fileWriter.append(evalRank.get(i).toString());
				fileWriter.append("\n");
				if(stdEval != null){
			        fileWriter.append("Evaluation rank std: ");
			        fileWriter.append(stdEval.get(i).toString());
					fileWriter.append("\n");
				}
		        fileWriter.append("Training rank: ");
		        fileWriter.append(trainRank.get(i).toString());
		        fileWriter.append("\n");
				if(stdEval != null){
			        fileWriter.append("Training rank std: ");
			        fileWriter.append(stdTrain.get(i).toString());
					fileWriter.append("\n");
				}
		        fileWriter.append("\n");
				fileWriter.flush();
			}
			fileWriter.close();
			
        } catch(IOException e){
            throw new RuntimeException(e);
        }
	}
	public void writeFeatureConfusions(ArrayList<INDArray> stdEval, ArrayList<INDArray> stdTrain){
		try {
            File write = new File(RESULTS_DIR + "FeatureConfusion-" + task + ".txt");
			write.delete();
			write.createNewFile();
			FileWriter fileWriter = new FileWriter(write);
			
			for(int i = 0; i < trainFeatureConf.size(); i++){
				fileWriter.append("Iteration: " + i);
				fileWriter.append("\n");
		        fileWriter.append("Evaluation feature confusion: ");
		        fileWriter.append(evalFeatureConf.get(i).toString());
				fileWriter.append("\n");
				if(stdEval != null){
			        fileWriter.append("Evaluation feature confusion std: ");
			        fileWriter.append(stdEval.get(i).toString());
					fileWriter.append("\n");
				}
		        fileWriter.append("Training feature confusion: ");
		        fileWriter.append(trainFeatureConf.get(i).toString());
		        fileWriter.append("\n");
				if(stdEval != null){
			        fileWriter.append("Training feature confusion std: ");
			        fileWriter.append(stdTrain.get(i).toString());
					fileWriter.append("\n");
				}
		        fileWriter.append("\n");
				fileWriter.flush();
			}
			fileWriter.close();
			
        } catch(IOException e){
            throw new RuntimeException(e);
        }
	}
	
	public ArrayList<Double> getEvalScore(){
		return evalScore;
	}
	
	public ArrayList<Double> getTrainScore(){
		return trainScore;
	}
	
	public ArrayList<Double> getEvalAccuracy(){
		return evalAccuracy;
	}
	
	public ArrayList<Double> getTrainAccuracy(){
		return trainAccuracy;
	}

	public ArrayList<INDArray> getTrainRank(){
		return trainRank;
	}

	public ArrayList<INDArray> getEvalRank(){
		return evalRank;
	}
	
	public ArrayList<INDArray> getEvalFeatureConf(){
		return evalFeatureConf;
	}
	
	public ArrayList<INDArray> getTrainFeatureConf(){
		return trainFeatureConf;
	}

	public void setEvalScore(ArrayList<Double> es){
		evalScore = es;
	}
	
	public void setTrainScore(ArrayList<Double> ts){
		trainScore = ts;
	}
	
	public void setEvalAccuracy(ArrayList<Double> ea){
		evalAccuracy = ea;
	}
	
	public void setTrainAccuracy(ArrayList<Double> et){
		trainAccuracy = et;
	}

	public void setTrainRank(ArrayList<INDArray> tr){
		trainRank = tr;
	}

	public void setEvalRank(ArrayList<INDArray> er){
		evalRank = er;
	}
	
	public void setEvalFeatureConf(ArrayList<INDArray> efc){
		evalFeatureConf = efc;
	}
	
	public void setTrainFeatureConf(ArrayList<INDArray> tfc){
		 trainFeatureConf = tfc;
	}
	
}
