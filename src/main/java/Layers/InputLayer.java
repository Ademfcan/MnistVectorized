package Layers;

import Activation.Relu;
import org.nd4j.linalg.api.ndarray.INDArray;


public class InputLayer extends Layer{
    public InputLayer(int inputShape) {
        super(new Relu(), null, inputShape, inputShape);
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
