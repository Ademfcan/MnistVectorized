package Activation;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class ActivationFunction {
    public abstract void initializeWeights(INDArray weights);
    public abstract INDArray func(INDArray v);
    public abstract INDArray derivative(INDArray v);

}
