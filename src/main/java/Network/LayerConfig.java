package Network;

import Learning.LearningRateProvider;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

public class LayerConfig {
    public String layerType; // "input", "dense", "output"
    public int shapeIn;
    public int shapeOut;
    public String activation;
    public LearningRateProvider learningRateProvider; // Should be serializable

    public String weightFile;
    public String biasFile;

    public LayerConfig() {}

    public LayerConfig(String type, int shapeIn, int shapeOut, String activation,
                       LearningRateProvider learningRateProvider, String weightFile, String biasFile) {
        this.layerType = type;
        this.shapeIn = shapeIn;
        this.shapeOut = shapeOut;
        this.activation = activation;
        this.learningRateProvider = learningRateProvider;
        this.weightFile = weightFile;
        this.biasFile = biasFile;
    }
}
