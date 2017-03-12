package org.deeplearning4j.nn.layers;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

import java.util.Arrays;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossCalculation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Required for handling the minibatch case and the different size of output for each sample in the minibatch
 * 
 * @author sorinMD
 *
 */
public class CustomOutputLayer extends BaseOutputLayer<org.deeplearning4j.nn.conf.layers.CustomOutputLayer>{
    
    /**
     * the number of classes for each sample. The size of this array also indicates the minibatch size.
     */
    protected INDArray numberOfActions;
    
    private double fullNetworkL1;
    private double fullNetworkL2;
    
	public CustomOutputLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public CustomOutputLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }
    
    public void setNumberOfActions(INDArray nAct){
    	numberOfActions = nAct;
    }
    
    public INDArray getNumberOfActions(){
    	return numberOfActions;
    }

    /** Compute score after labels and input have been set.
     * @param fullNetworkL1 L1 regularization term for the entire network
     * @param fullNetworkL2 L2 regularization term for the entire network
     * @param training whether score should be calculated at train or test time (this affects things like application of
     *                 dropout, etc)
     * @return score (loss function)
     */
    public double computeScore( double fullNetworkL1, double fullNetworkL2, boolean training) {
        if( input == null || labels == null )
            throw new IllegalStateException("Cannot calculate score without input and labels");
        this.fullNetworkL1 = fullNetworkL1;
        this.fullNetworkL2 = fullNetworkL2;
        INDArray preOut = preOutput2d(training);
        //we need to compute the output for each sample separately due to this layer being over multiple samples
        LossFunctions.LossFunction lf = ((org.deeplearning4j.nn.conf.layers.BaseOutputLayer)conf.getLayer()).getLossFunction();
        if ( (lf == LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD || lf == LossFunctions.LossFunction.MCXENT) && layerConf().getActivationFunction().equals("softmax")) {
            //special case: softmax + NLL or MCXENT: use log softmax to avoid numerical underflow
            setScore(null,preOut);
        } else {
        	INDArray output = output(training);
            setScoreWithZ(output);
        }
        return score;
    }
    
    @Override
    public void computeGradientAndScore() {
        if(input == null || labels == null)
            return;
        
        INDArray preOut = preOutput2d(true);
        Triple<Gradient,INDArray,INDArray> triple = getGradientsAndDelta(preOut);
        this.gradient = triple.getFirst();
        setScore(triple.getThird(), preOut);
    }

    @Override
    protected void setScoreWithZ(INDArray z) {
    	setScore(z, null);
    }
    
    private void setScore(INDArray z, INDArray preOut ){
        if (layerConf().getLossFunction() == LossFunctions.LossFunction.CUSTOM) {
            LossFunction create = Nd4j.getOpFactory().createLossFunction(layerConf().getCustomLossFunction(), input, z);
            create.exec();
            score = create.getFinalResult().doubleValue();
        }
        else {
        	//NOTE: due to the weird minibatch implementation where each sample is on its own a minibatch, we need to iterate over each sample separately
        	int ind = numberOfActions.getRow(0).getInt(0);
        	INDArray preOutput2 = preOut.get(NDArrayIndex.interval(0,ind),NDArrayIndex.all());
        	INDArray labels2 = getLabels2d().get(NDArrayIndex.interval(0,ind),NDArrayIndex.all());
        	
        	score = LossCalculation.builder()
                    .l1(fullNetworkL1).l2(fullNetworkL2)
                    .labels(labels2).z(z)
                    .preOut(preOutput2).activationFn(conf().getLayer().getActivationFunction())
                    .lossFunction(layerConf().getLossFunction())
                    .miniBatch(conf.isMiniBatch()).miniBatchSize(numberOfActions.getRow(0).getInt(0))
                    .useRegularization(conf.isUseRegularization())
                    .mask(maskArray).build().score();
        	
        	for(int i = 1; i< numberOfActions.size(0); i++){
        		preOutput2 = preOut.get(NDArrayIndex.interval(ind,ind + numberOfActions.getRow(i).getInt(0)),NDArrayIndex.all());
        		labels2 = getLabels2d().get(NDArrayIndex.interval(ind,ind + numberOfActions.getRow(i).getInt(0)),NDArrayIndex.all());
        		ind += numberOfActions.getRow(i).getInt(0);
            	score += LossCalculation.builder()
                        .l1(fullNetworkL1).l2(fullNetworkL2)
                        .labels(labels2).z(z)
                        .preOut(preOutput2).activationFn(conf().getLayer().getActivationFunction())
                        .lossFunction(layerConf().getLossFunction())
                        .miniBatch(conf.isMiniBatch()).miniBatchSize(numberOfActions.getRow(i).getInt(0))
                        .useRegularization(conf.isUseRegularization())
                        .mask(maskArray).build().score();
        	}
        	if(conf.isMiniBatch())
        		score /= (double) getNumberOfActions().size(0);
        }
    }
    
    @Override
    public Pair<Gradient,INDArray> backpropGradient(INDArray epsilon) {
    	Triple<Gradient,INDArray,INDArray> triple = getGradientsAndDelta(preOutput2d(true));	//Returns Gradient and delta^(this), not Gradient and epsilon^(this-1)
    	INDArray delta = triple.getSecond();

    	INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(delta.transpose()).transpose();
        return new Pair<>(triple.getFirst(),epsilonNext);
    }
    
    
    /** Returns tuple: {Gradient,Delta,Output} given preOut */
    private Triple<Gradient,INDArray,INDArray> getGradientsAndDelta(INDArray preOut){
    	int ind = numberOfActions.getRow(0).getInt(0);
    	INDArray preOutput2 = preOut.get(NDArrayIndex.interval(0,ind),NDArrayIndex.all());
    	INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), preOutput2.dup()));
    	for(int i = 1; i< numberOfActions.size(0); i++){
    		preOutput2 = preOut.get(NDArrayIndex.interval(ind,ind + numberOfActions.getRow(i).getInt(0)),NDArrayIndex.all());
    		ind += numberOfActions.getRow(i).getInt(0);
    		output = Nd4j.vstack(output,Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), preOutput2.dup())));
    	}
    	
    	INDArray outSubLabels = output.sub(getLabels2d());
    	Gradient gradient = new DefaultGradient();
    	
        INDArray weightGradView = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);
        INDArray biasGradView = gradientViews.get(DefaultParamInitializer.BIAS_KEY);

        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY,weightGradView);
        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY,biasGradView);

        if(maskArray != null){
            //Masking on gradients. Mask values are 0 or 1. If 0: no output -> no error for this example
            outSubLabels.muliColumnVector(maskArray);
        }
    	
        Triple<Gradient,INDArray,INDArray> triple;
        switch (layerConf().getLossFunction()) {
            case NEGATIVELOGLIKELIHOOD:
            case MCXENT:	//cross-entropy (multi-class, with one-hot encoding)
                Nd4j.gemm(input,outSubLabels,weightGradView,true,false,1.0,0.0);    //Equivalent to:  weightGradView.assign(input.transpose().mmul(outSubLabels));
                biasGradView.assign(outSubLabels.sum(0));   //TODO: do this without the assign
                triple = new Triple<>(gradient,outSubLabels,output);
                break;

            case XENT: // cross-entropy (single binary output variable)
                Nd4j.gemm(input, outSubLabels.div(output.mul(output.rsub(1))), weightGradView, true, false, 1.0, 0.0);  //Equivalent to:  weightGradView.assign(input.transpose().mmul(outSubLabels.div(output.mul(output.rsub(1)))));
                biasGradView.assign(outSubLabels.sum(0));    //TODO: do this without the assign
                triple = new Triple<>(gradient,outSubLabels,output);
                break;

            case MSE: // mean squared error
                INDArray delta = outSubLabels.mul(derivativeActivation(preOut));
                Nd4j.gemm(input,delta,weightGradView,true,false,1.0,0.0);   //Equivalent to:  weightGradView.assign(input.transpose().mmul(delta));
                biasGradView.assign(delta.sum(0));         //TODO: do this without the assign
                triple = new Triple<>(gradient,delta,output);
                break;

            case EXPLL: // exponential logarithmic
                Nd4j.gemm(input,labels.rsub(1).divi(output),weightGradView,true,false,1.0,0.0); //Equivalent to:  weightGradView.assign(input.transpose().mmul(labels.rsub(1).divi(output)));
                biasGradView.assign(outSubLabels.sum(0));   //TODO: do this without the assign
                triple = new Triple<>(gradient,outSubLabels,output);
                break;

            case RMSE_XENT: // root mean squared error cross entropy
                INDArray squaredrmseXentDiff = pow(outSubLabels, 2.0);
                INDArray sqrt = sqrt(squaredrmseXentDiff);
                Nd4j.gemm(input,sqrt,weightGradView,true,false,1.0,0.0);    //Equivalent to: weightGradView.assign(input.transpose().mmul(sqrt));
                biasGradView.assign(outSubLabels.sum(0));   //TODO: do this without the assign
                triple = new Triple<>(gradient,outSubLabels,output);
                break;

            case SQUARED_LOSS:
                Nd4j.gemm(input,outSubLabels.mul(outSubLabels),weightGradView,true,false,1.0,0.0);  //Equivalent to: weightGradView.assign(input.transpose().mmul(outSubLabels.mul(outSubLabels)));
                biasGradView.assign(outSubLabels.sum(0));   //TODO: do this without the assign
                triple = new Triple<>(gradient,outSubLabels,output);
                break;
            default:
                throw new IllegalStateException("Invalid loss function: " + layerConf().getLossFunction());
        }

        return triple;
    }
    
    /**
     * Classify input
     * @param training determines if its training
     * the input (can either be a matrix or vector)
     * If it's a matrix, each row is considered an example
     * and associated rows are classified accordingly.
     * Each row will be the likelihood of a label given that example
     * @return a probability distribution for each row
     */
    public  INDArray output(boolean training) {
        if(input == null)
            throw new IllegalArgumentException("No null input allowed");
        INDArray preOutput = preOutput2d(training);
        if(conf.getLayer().getActivationFunction().equals("softmax")) {
        	int ind = numberOfActions.getRow(0).getInt(0);
        	INDArray preOutput2 = preOutput.get(NDArrayIndex.interval(0,ind),NDArrayIndex.all());
            
        	//do a softmax over each and stack the results to handle the minibatch case
        	INDArray out = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), preOutput2.dup()));
        	for(int i = 1; i < numberOfActions.size(0); i++){
        		preOutput2 = preOutput.get(NDArrayIndex.interval(ind,ind + numberOfActions.getRow(i).getInt(0)),NDArrayIndex.all());
        		ind += numberOfActions.getRow(i).getInt(0);
                INDArray z = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), preOutput2.dup()));
                out = Nd4j.vstack(out,z);
        	}

            if(maskArray != null){
                out.muliColumnVector(maskArray);
            }
        	
            return out;
        }

        //I am not sure what this is doing here as dropout was applied in preoutput so nothing will happen.....
        if(training)
            applyDropOutIfNecessary(training);

        return super.activate(true);
    }
    
    @Override
    public void setMaskArray(INDArray maskArray) {
    	//TODO: not sure if I should modify the mask here, hence why I added this method
        this.maskArray = maskArray;
    }
    
}
