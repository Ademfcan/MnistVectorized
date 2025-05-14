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

import java.io.File;
import java.io.IOException;

public class train {
    public static void main(String[] args) throws IOException {

        LearningRateProvider constant = new ConstantLearningRate(0.015);
        NNetwork network = new NNetwork(
                new InputLayer(28 * 28),
                new OutputLayer(constant, 512, 10),
                new Layer(ActivationFunction.RELU, constant, 28 * 28, 512)
        );


        MnistLoader loaderTrain = new MnistLoader("trainingdata/emnist-digits-train-images-idx3-ubyte",
                "trainingdata/emnist-digits-train-labels-idx1-ubyte");
        MnistLoader loaderTest = new MnistLoader("trainingdata/emnist-digits-test-images-idx3-ubyte",
                "trainingdata/emnist-digits-test-labels-idx1-ubyte");
        int batchSize = 64;
        EndCondition maxEpochs = new EpochEndCondition(1);
        BatchedTrainer trainer = new BatchedTrainer(network, loaderTrain, loaderTest, batchSize, maxEpochs);

        trainer.train();

        network.save(new File("models/Out512.zip"));
    }
}