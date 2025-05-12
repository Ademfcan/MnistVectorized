package Layers;

import Activation.ActivationFunction;
import Cost.CostFunction;
import Learning.LearningRateProvider;
import org.nd4j.linalg.api.ndarray.INDArray;

public class OutputLayer extends Layer{
    CostFunction costFunction;
    public OutputLayer(CostFunction costFunction, ActivationFunction activationFunc, LearningRateProvider learningRateProvider, int weightR, int weightC) {
        super(activationFunc, learningRateProvider, weightR, weightC);
        this.costFunction = costFunction;
    }

    @Override
    public INDArray forward(){
        // handle output activation function and cost function
    }

    @Override
    public INDArray backward(){
        // backprop starts here
    }
}
