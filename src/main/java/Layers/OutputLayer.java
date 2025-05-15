package Layers;

import Activation.ActivationFunction;
import Learning.LearningRateProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class OutputLayer extends Layer{

    public OutputLayer(LearningRateProvider learningRateProvider, int layerSize) {
        super(ActivationFunction.SOFTMAXX, learningRateProvider, layerSize);
    }

    public void backward(INDArray expectedOutputs) {
        int batchSize = expectedOutputs.columns();

        errorSignal = activations.sub(expectedOutputs);

        INDArray dLdW = errorSignal.mmul(previous.activations.transpose()).div(batchSize);
        INDArray dLdB = errorSignal.sum(1).reshape(errorSignal.rows(), 1).div(batchSize);

        double learningRate = learningRateProvider.getLearningRate(this);
        weights.subi(dLdW.mul(learningRate));
        biases.subi(dLdB.mul(learningRate));

        previous.backward(errorSignal, weights);
    }

    public INDArray getError(int[] correctLabelId){
        INDArray error = Nd4j.zeros(correctLabelId.length);

        for (int i = 0; i < correctLabelId.length; i++) {
            int classIdx = correctLabelId[i];
            // (row, col) == (class, i)
            double prob = activations.getDouble(classIdx, i);
            error.putScalar(i, -Math.log(prob + 1e-10)); // add epsilon for numerical stability
        }

        return error;
    }



}
