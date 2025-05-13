package Activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public enum ActivationFunction {
    RELU{
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
    },
    SOFTMAXX {
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
    };

    public abstract void initializeWeights(INDArray weights);
    public abstract INDArray func(INDArray v);
    public abstract INDArray derivative(INDArray v);
}
