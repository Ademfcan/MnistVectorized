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


    public void save(File zipFile) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        List<LayerConfig> layerConfigs = new ArrayList<>();

        FileOutputStream fos = new FileOutputStream(zipFile);
        ZipOutputStream zipOut = new ZipOutputStream(fos);

        int layerIndex = 0;

        // Save input layer config (no weights/biases)
        layerConfigs.add(new LayerConfig("input",
                inputLayer.getShapeIn(),
                inputLayer.getShapeOut(),
                null,
                null,
                null,
                null
        ));

        // Save hidden layers
        for (Layer layer : hiddenLayers) {
            String weightFile = "layer" + layerIndex + "_weights.bin";
            String biasFile = "layer" + layerIndex + "_biases.bin";

            writeINDArrayToZip(zipOut, weightFile, layer.getWeights());
            writeINDArrayToZip(zipOut, biasFile, layer.getBiases());

            layerConfigs.add(new LayerConfig("dense",
                    layer.getShapeIn(),
                    layer.getShapeOut(),
                    layer.getActivationFunc().toString(),
                    layer.getLearningRateProvider(),
                    weightFile,
                    biasFile
            ));

            layerIndex++;
        }

        // Save output layer
        String weightFile = "output_weights.bin";
        String biasFile = "output_biases.bin";

        writeINDArrayToZip(zipOut, weightFile, outputLayer.getWeights());
        writeINDArrayToZip(zipOut, biasFile, outputLayer.getBiases());

        layerConfigs.add(new LayerConfig("output",
                outputLayer.getShapeIn(),
                outputLayer.getShapeOut(),
                outputLayer.getActivationFunc().toString(),
                outputLayer.getLearningRateProvider(),
                weightFile,
                biasFile
        ));

        // Save model config as JSON
        ZipEntry configEntry = new ZipEntry("model.json");
        zipOut.putNextEntry(configEntry);
        zipOut.write(mapper.writeValueAsBytes(layerConfigs));
        zipOut.closeEntry();

        zipOut.close();
        fos.close();
    }

    private void writeINDArrayToZip(ZipOutputStream zipOut, String name, INDArray arr) throws IOException {
        ZipEntry entry = new ZipEntry(name);
        zipOut.putNextEntry(entry);
        DataOutputStream dos = new DataOutputStream(zipOut);
        Nd4j.write(arr, dos);
        zipOut.closeEntry();
    }

    public static NNetwork fromZip(File zipFile) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        List<LayerConfig> configs;
        INDArray[] weightsAndBiases;

        try (ZipFile zip = new ZipFile(zipFile)) {
            ZipEntry configEntry = zip.getEntry("model.json");
            InputStream configStream = zip.getInputStream(configEntry);
            configs = mapper.readValue(configStream,
                    mapper.getTypeFactory().constructCollectionType(List.class, LayerConfig.class));

            weightsAndBiases = new INDArray[configs.size() * 2]; // each layer (except input) has weights and biases

            int index = 0;
            for (LayerConfig cfg : configs) {
                if (cfg.weightFile != null && cfg.biasFile != null) {
                    weightsAndBiases[index++] = Nd4j.read(zip.getInputStream(zip.getEntry(cfg.weightFile)));
                    weightsAndBiases[index++] = Nd4j.read(zip.getInputStream(zip.getEntry(cfg.biasFile)));
                }
            }
        }

        // Construct actual layers from configs (mock example — you'll need factory logic here)
        InputLayer inputLayer = new InputLayer(configs.get(0).shapeOut);
        List<Layer> hidden = new ArrayList<>();
        int wi = 0;

        for (int i = 1; i < configs.size() - 1; i++) {
            LayerConfig cfg = configs.get(i);
            Layer layer = new Layer(ActivationFunction.valueOf(cfg.activation), cfg.learningRateProvider, cfg.shapeIn, cfg.shapeOut);
            layer.setWeights(weightsAndBiases[wi++]);
            layer.setBiases(weightsAndBiases[wi++]);
            hidden.add(layer);
        }

        LayerConfig outCfg = configs.get(configs.size() - 1);
        OutputLayer outputLayer = new OutputLayer(outCfg.learningRateProvider, outCfg.shapeIn, outCfg.shapeOut);
        outputLayer.setWeights(weightsAndBiases[wi++]);
        outputLayer.setBiases(weightsAndBiases[wi]);

        return new NNetwork(inputLayer, outputLayer, hidden.toArray(new Layer[0]));
    }





}
