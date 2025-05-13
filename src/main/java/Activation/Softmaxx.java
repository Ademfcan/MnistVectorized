package Activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Softmaxx extends ActivationFunction{

    @Override
    public void initializeWeights(INDArray weights) {
        Initalization.XavierInit(weights);
    }

    @Override
    public INDArray func(INDArray v) {
        INDArray v_stable = v.sub(v.max());
        return Transforms.exp(v_stable).div(Transforms.exp(v_stable).sum(0));
    }

    @Override
    public INDArray derivative(INDArray v) {
        // wont be used
        return null;
    }

}
