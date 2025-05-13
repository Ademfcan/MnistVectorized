package Network;

import Layers.InputLayer;
import Layers.Layer;
import Layers.OutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class NNetwork {
    public final InputLayer inputLayer;
    public final OutputLayer outputLayer;
    public final Layer[] hiddenLayers;

    public NNetwork(InputLayer inputLayer, OutputLayer outputLayer, Layer... hiddenLayers){
        if(!assertShape(inputLayer, outputLayer, hiddenLayers)){
            throw new IllegalArgumentException("All layer shapes must match!");
        }

        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.hiddenLayers = hiddenLayers;
        setConnections();
    }

    public boolean assertShape(InputLayer inputLayer, OutputLayer outputLayer, Layer[] hiddenLayers){
        boolean match = true;

        for(int i = 1; i<hiddenLayers.length; i++){
            match = match && (hiddenLayers[i-1].getShapeOut() == hiddenLayers[i].getShapeIn());
        }

        return match && (hiddenLayers[hiddenLayers.length-1].getShapeOut() == outputLayer.getShapeIn());
    }

    public void setConnections(){
        inputLayer.setNext(hiddenLayers[0]);
        hiddenLayers[0].setPrevious(inputLayer);

        for(int i = 1; i<hiddenLayers.length; i++){
            hiddenLayers[i-1].setNext(hiddenLayers[i]);
            hiddenLayers[i].setPrevious(hiddenLayers[i-1]);
        }

        hiddenLayers[hiddenLayers.length-1].setNext(outputLayer);
        outputLayer.setPrevious(hiddenLayers[hiddenLayers.length-1]);
    }

    public void runInference(INDArray input){
        inputLayer.forward(input);
    }

    public void runBackprop(int[] correctLabelIds){
        outputLayer.backward(correctLabelIds);
    }


}
