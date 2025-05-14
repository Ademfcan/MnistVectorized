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
import java.util.Objects;

public class train {
    public static void main(String[] args) throws IOException {
        int batchSize = 64;
        int numEpochs = 20;

        LearningRateProvider constant = new ConstantLearningRate(0.015);
        NNetwork network = new NNetwork(
                new InputLayer(28 * 28),
                new OutputLayer(constant, 512, 10),
                new Layer(ActivationFunction.RELU, constant, 28 * 28, 512)
        );

//        NNetwork network = NNetwork.fromZip(new File(Objects.
//                requireNonNull(main.class.getClassLoader().getResource("out.zip")).getPath()));

        MnistLoader loaderTrain = new MnistLoader(true);
        MnistLoader loaderTest = new MnistLoader(false);
        EndCondition maxEpochs = new EpochEndCondition(numEpochs);

        BatchedTrainer trainer = new BatchedTrainer(network, loaderTrain, loaderTest, batchSize, maxEpochs);
        trainer.train();

        network.save(new File("Out512.zip"));
    }
}