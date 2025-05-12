package Learning;

import Layers.Layer;

public class ConstantLearningRate extends LearningRateProvider{
    double learningRate;
    public ConstantLearningRate(double learningRate){
        this.learningRate = learningRate;
    }

    @Override
    public double getLearningRate(Layer layer) {
        return learningRate;
    }
}
