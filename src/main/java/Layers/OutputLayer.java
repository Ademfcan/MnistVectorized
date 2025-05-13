package Layers;

import Activation.Softmaxx;
import Learning.LearningRateProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class OutputLayer extends Layer{

    public OutputLayer(LearningRateProvider learningRateProvider, int shapeIn, int shapeOut) {
        super(new Softmaxx(), learningRateProvider, shapeIn, shapeOut);
    }

    public void backward(int[] correctLabels){
        int batchSize = correctLabels.length;

        INDArray oneHot = Nd4j.zerosLike(activations);
        for(int i = 0; i<batchSize;i++){
            oneHot.putScalar(correctLabels[i], i, 1.0f);
        }
        errorSignal = activations.sub(oneHot);

        INDArray dLdW = errorSignal.mmul(previous.activations.transpose()).div(batchSize);
        INDArray dLdB = errorSignal.sum(1).reshape(errorSignal.rows(), 1).div(batchSize);
        dLdWs.addi(dLdW);
        dLdBs.addi(dLdB);


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
