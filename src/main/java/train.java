import Activation.ActivationFunction;
import Data.MnistLoader;
import Layers.InputLayer;
import Layers.Layer;
import Layers.OutputLayer;
import Learning.ConstantLearningRate;
import Learning.LearningRateProvider;
import Network.NNetwork;
import Training.BatchedTrainer;
import Training.EndCondition;
import Training.EpochEndCondition;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;
import java.util.function.Function;

public class train {
    /**
     * How the default model was created
     */
    public static void main(String[] args) throws IOException {
        // constant learning rate of 0.015
        LearningRateProvider constant = new ConstantLearningRate(0.015);

        // take in flattened 28x28 input,
        // run through hidden layers,
        // output 10 clases for numbers 0-9
        NNetwork network = new NNetwork(
                new InputLayer(28 * 28),
                new OutputLayer(constant, 10),
                new Layer(ActivationFunction.RELU, constant, 128),
                new Layer(ActivationFunction.RELU, constant, 128)
        );
        // mnist training data included by default
        MnistLoader loaderTrain = new MnistLoader(true);
        MnistLoader loaderTest = new MnistLoader(false);

        int batchSize = 64;
        // model was trained on 50 epochs
        EndCondition maxEpochs = new EpochEndCondition(1);
        BatchedTrainer trainer = new BatchedTrainer(network, loaderTrain,
                loaderTest, batchSize, maxEpochs);

        // train the network
        trainer.train();

        // save output
        network.save(new File("out.zip"));
    }
}