package org.deeplearning4j.nn.conf.layers;

import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;

/**
 * 
 * @author sorinMD
 *
 */
@Data
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class CustomOutputLayer extends BaseOutputLayer{

	public CustomOutputLayer(){
		super();
	}
	
    protected CustomOutputLayer(Builder builder) {
    	super(builder);
    }

    @NoArgsConstructor
    public static class Builder extends BaseOutputLayer.Builder<Builder> {

        public Builder(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
        }
        
        @Override
        @SuppressWarnings("unchecked")
        public CustomOutputLayer build() {
            return new CustomOutputLayer(this);
        }
    }

}
