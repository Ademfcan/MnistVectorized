package Learning;

import Layers.Layer;
import com.fasterxml.jackson.annotation.JsonTypeName;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonTypeName("constant")
public class ConstantLearningRate implements LearningRateProvider {
    @JsonProperty("learningRate")
    private final double learningRate;

    @JsonCreator
    public ConstantLearningRate(@JsonProperty("learningRate") double learningRate) {
        this.learningRate = learningRate;
    }
    @Override
    public double getLearningRate(Layer layer) {
        return learningRate;
    }
}
