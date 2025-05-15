package Network;

import Activation.ActivationFunction;
import Layers.InputLayer;
import Layers.Layer;
import Layers.OutputLayer;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

public class NNetwork {
    public final InputLayer inputLayer;
    public final OutputLayer outputLayer;
    public final Layer[] hiddenLayers;

    public NNetwork(InputLayer inputLayer, OutputLayer outputLayer, Layer... hiddenLayers){
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.hiddenLayers = hiddenLayers;
        connectShapes();
        setConnections();
    }

    private void connectShapes() {
        hiddenLayers[0].initInputShape(inputLayer.getShapeOut());
        for(int i = 1; i<hiddenLayers.length; i++){
            hiddenLayers[i].initInputShape(hiddenLayers[i-1].getShapeOut());
        }
        outputLayer.initInputShape(hiddenLayers[hiddenLayers.length-1].getShapeOut());
    }

    private void setConnections(){
        inputLayer.setNext(hiddenLayers[0]);
        hiddenLayers[0].setPrevious(inputLayer);

        for(int i = 1; i<hiddenLayers.length; i++){
            hiddenLayers[i-1].setNext(hiddenLayers[i]);
            hiddenLayers[i].setPrevious(hiddenLayers[i-1]);
        }

        hiddenLayers[hiddenLayers.length-1].setNext(outputLayer);
        outputLayer.setPrevious(hiddenLayers[hiddenLayers.length-1]);
    }

    /**
     * Each layer, apart from the input layer, has weights and biases.
     * The number of weights for each layer is the shapeIn*shapeOut (matrix), and
     * the number of biases for each layer is shapeOut
     */
    public double getNumParams(){
        double count = 0;
        for(Layer layer : hiddenLayers){
            count += layer.getShapeIn() * layer.getShapeOut(); // weights
            count += layer.getShapeOut(); // biases
        }

        count += outputLayer.getShapeIn() * outputLayer.getShapeOut(); // weights
        count += outputLayer.getShapeOut(); // biases
        return count;
    }

    public void runInference(INDArray input){
        inputLayer.forward(input);
    }

    public void runBackprop(INDArray expectedOutputs){
        outputLayer.backward(expectedOutputs);
    }

    public void save(File outputFile) throws IOException {
        ZipHelper.save(this, outputFile);
    }








}
