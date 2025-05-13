package Learning;

import Layers.Layer;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = ConstantLearningRate.class, name = "constant"),
})
public interface LearningRateProvider {
    double getLearningRate(Layer layer);
}
