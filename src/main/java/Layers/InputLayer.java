package Layers;

import Activation.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;


public class InputLayer extends Layer{
    public InputLayer(int inputShape) {
        super(ActivationFunction.RELU, null, inputShape, inputShape);
    }

    @Override
    public void forward(INDArray networkInput){
        // forward prop starts here
        this.activations = networkInput;
        next.forward(this.activations);

    }

    @Override
    public void backward(INDArray nextErrorSignal, INDArray nextWeights){
        return;
    }

}
