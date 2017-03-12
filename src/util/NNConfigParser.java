package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Configuration parser.
 * 
 * @author sorinMD
 *
 */
public class NNConfigParser {
	public static String CONFIG_FILE = "training_config.txt";
	private double learningRate;
	private int minibatchSize;
	private int epochs;
	private double labelWeight;
	private double metricWeight;
	private int task;
	private int preTrainTask;
	private int preTrainEpochs;
	private String dataType;
	private String path;
	private int samples;
	private boolean normalisation;
	
	public void parseConfig(String filePath){
		double[] params = new double[5];
		BufferedReader reader = null;
		if(filePath == null){
			try {
				reader = new BufferedReader(new FileReader(new File(CONFIG_FILE)));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}else{
			try {
				reader = new BufferedReader(new FileReader(new File(filePath)));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}
		 String nextLine = null;
		try {
			nextLine = reader.readLine();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	        while (nextLine != null) {
	            if (nextLine.startsWith("LearningRate")) {
	                String p[] = nextLine.split("=");
	                learningRate = Double.parseDouble(p[1]);
	            }else if (nextLine.startsWith("MiniBatchSize")) {
	                String p[] = nextLine.split("=");
	                minibatchSize = Integer.parseInt(p[1]);
	            }else if (nextLine.startsWith("Epochs")) {
	                String p[] = nextLine.split("=");
	                epochs = Integer.parseInt(p[1]);
	            }else if (nextLine.startsWith("LabelWeight")) {
	                String p[] = nextLine.split("=");
	                labelWeight = Double.parseDouble(p[1]);
	            }else if (nextLine.startsWith("MetricWeight")) {
	                String p[] = nextLine.split("=");
	                metricWeight = Double.parseDouble(p[1]);
	            }else if (nextLine.startsWith("Task")) {
	                String p[] = nextLine.split("=");
	                task = Integer.parseInt(p[1]);
	            }else if (nextLine.startsWith("DataType")) {
	                String p[] = nextLine.split("=");
	                dataType = p[1];
	            }else if (nextLine.startsWith("Path")) {
	                String p[] = nextLine.split("=");
	                path = p[1];
	            }else if (nextLine.startsWith("Samples")) {
	                String p[] = nextLine.split("=");
	                samples = Integer.parseInt(p[1]);
	            }else if (nextLine.startsWith("Normalisation")) {
	                String p[] = nextLine.split("=");
	                normalisation = Boolean.parseBoolean(p[1]);
	            }else if (nextLine.startsWith("PreTrainTask")) {
	                String p[] = nextLine.split("=");
	                preTrainTask = Integer.parseInt(p[1]);
	            }else if (nextLine.startsWith("PreTrainEpochs")) {
	                String p[] = nextLine.split("=");
	                preTrainEpochs = Integer.parseInt(p[1]);
	            }
	            else{
	            	throw new RuntimeException("Cannot read configuration: unknown parameter");
	            }
	        	try {
					nextLine = reader.readLine();
				} catch (IOException e) {
					e.printStackTrace();
				}
	        }
		
	}
	
	public NNConfigParser(String filePath){
		parseConfig(filePath);
	}
	
	public double getLearningRate(){
		return learningRate;
	}

	public double getLabelWeight(){
		return labelWeight;
	}
	
	public double getMetricWeight(){
		return metricWeight;
	}
	
	public int getEpochs(){
		return epochs;
	}
	
	public int getMiniBatchSize(){
		return minibatchSize;
	}
	
	public int getTask(){
		return task;
	}
	
	public String getDataType(){
		return dataType;
	}
	
	public String getDataPath(){
		return path;
	}
	
	public int getNumberOfSamples(){
		return samples;
	}
	
	public boolean getNormalisation(){
		return normalisation;
	}

	public int getPreTrainTask() {
		return preTrainTask;
	}

	public int getPreTrainEpochs() {
		return preTrainEpochs;
	}

	
}
