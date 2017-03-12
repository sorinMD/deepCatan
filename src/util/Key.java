package util;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * The key structure to use a row INDArray as a key in a map.
 * 
 * @author sorinMD
 *
 */
public class Key {
	public final INDArray array;
	
	public Key(INDArray a) {
		array = a;
	}
	
	@Override
	public boolean equals(Object obj) {
		if(obj instanceof Key){
			return this.hashCode() == obj.hashCode();
		}
		return false;
	}
	
	@Override
	public int hashCode() {
        int result = 17;
        //iterate over all elements
        for(int i = 0; i < array.size(1); i ++){
        	result = 31 * result + array.getInt(i);
        }
        return result;
	}
	
}
