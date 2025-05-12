package Network;

import Cost.CostFunction;
import Layers.InputLayer;
import Layers.Layer;
import Layers.OutputLayer;

public class NNetwork {
    public NNetwork(InputLayer inputLayer, OutputLayer outputLayer, Layer... layers){
        if(!assertShape(inputLayer, outputLayer, layers)){
            throw new IllegalArgumentException("All layer shapes must match!");
        }
    }

    public boolean assertShape(InputLayer inputLayer, OutputLayer outputLayer, Layer[] layers){
        boolean match = (inputLayer.getWeightC() == layers[0].getWeightR());

        for(int i = 1; i<layers.length; i++){
            match = match && (layers[i-1].getWeightC() == layers[i].getWeightR());
        }

        return match && (layers[layers.length-1].getWeightC() == outputLayer.getWeightR());
    }



}
