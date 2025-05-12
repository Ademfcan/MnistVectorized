package Layers;

import Activation.ActivationFunction;
import Learning.LearningRateProvider;
import org.nd4j.linalg.api.ndarray.INDArray;

public class InputLayer extends Layer{
    public InputLayer(ActivationFunction activationFunc, LearningRateProvider learningRateProvider, int weightR, int weightC) {
        super(activationFunc, learningRateProvider, weightR, weightC);
    }

    @Override
    public INDArray forward(){
        // forward prop starts here
    }

    @Override
    public INDArray backward(){
        // backprop ends here
    }
}
