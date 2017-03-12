package model;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.CustomOutputLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.CatanMlp;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * The Mlp configuration used in all the experiments.
 * 
 * @author sorinMD
 *
 */
public class CatanMlpConfig{

    private int inputNum;
    private int outputNum;
    private long seed;
    private int iterations;
    private Updater updater;
    private double learningRate;
    private LossFunctions.LossFunction lossFunct = LossFunction.MSE;
    private OptimizationAlgorithm optAlg = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT; //default
    private WeightInit weightInit = WeightInit.XAVIER;
    private MultiLayerConfiguration conf;
    
    public CatanMlpConfig(int in, int out, long seed, int iterations, WeightInit wi, Updater u, double lr, LossFunctions.LossFunction lf, OptimizationAlgorithm oa) {
        this.inputNum = in;
        this.outputNum = out;
        this.seed = seed;
        this.iterations = iterations;
        this.weightInit = wi;
        this.updater = u;
        this.learningRate = lr;
        this.lossFunct = lf;
        this.optAlg = oa;
    }

    public CatanMlp init() {
        conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(optAlg)
                .learningRate(learningRate)
                .rmsDecay(0.9)
                .list(3)
                .layer(0, new DenseLayer.Builder()
				                .nIn(inputNum)
				                .nOut(256)
				                .activation("sigmoid")
				                .weightInit(weightInit)
				                .updater(updater)
				                .build())
                .layer(1, new DenseLayer.Builder()
				                .nIn(256)
				                .nOut(256)
				                .activation("relu")
				                .weightInit(weightInit)
				                .updater(updater)
				                .build())
                .layer(2, new CustomOutputLayer.Builder(lossFunct)
                                .nIn(256) // # input nodes
                                .nOut(outputNum) // # output nodes
                                .activation("softmax")
                                .weightInit(weightInit)
                                .updater(updater)
                                .build()
                )
                .backprop(true).pretrain(false)
                .build();

        CatanMlp model = new CatanMlp(conf);
        model.init();
        
        return model;
    }

    public MultiLayerConfiguration getNNConf(){
    	return conf;
    }
    
}
