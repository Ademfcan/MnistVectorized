package Learning;

import Layers.Layer;

public abstract class LearningRateProvider {
    public abstract double getLearningRate(Layer layer);
}
