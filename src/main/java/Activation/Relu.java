package Activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Relu extends ActivationFunction{
    private RectifiedLinear relu;
    public Relu(){
        relu = new RectifiedLinear();
    }

    @Override
    public void initializeWeights(INDArray weights) {
        Initalization.HeInit(weights);
    }

    @Override
    public INDArray func(INDArray v) {
        return Transforms.relu(v);
    }

    @Override
    public INDArray derivative(INDArray v) {
        return Transforms.step(v);
    }

}




