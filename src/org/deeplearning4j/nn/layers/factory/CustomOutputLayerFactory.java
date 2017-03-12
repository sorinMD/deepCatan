package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.CustomOutputLayer;
import org.deeplearning4j.nn.conf.layers.Layer;

public class CustomOutputLayerFactory extends DefaultLayerFactory{

	public CustomOutputLayerFactory(Class<? extends Layer> layerConfig) {
        super(layerConfig);
    }

    @Override
    protected org.deeplearning4j.nn.api.Layer getInstance(
    		NeuralNetConfiguration conf) {
        if(layerConfig instanceof CustomOutputLayer)
            return new org.deeplearning4j.nn.layers.CustomOutputLayer(conf);
    	//not sure if I should allow this....
    	return super.getInstance(conf);
    }

}
