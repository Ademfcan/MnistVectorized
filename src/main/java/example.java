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

public class example {
    /**
     * Example method for training on the emnist-digits dataset,
     * with custom preprocessing to threshold the float inputs of 0-1 by 0.5
     * [0-1] -> 0 if <= 0.5 or 1 if > 0.5
     */
    public static void main(String[] args) throws IOException {
        // constant learning rate of 0.015
        LearningRateProvider constant = new ConstantLearningRate(0.015);

        // take in flattened 28x28 input,
        // run through hidden layers,
        // output 10 clases for numbers 0-9
        NNetwork network = new NNetwork(
                new InputLayer(28 * 28),
                new OutputLayer(constant, 128, 10),
                new Layer(ActivationFunction.RELU, constant, 28 * 28, 128),
                new Layer(ActivationFunction.RELU, constant, 128, 128)
        );

        // binarization of data with threshold of 0.5
        // x < 0.5 = 0
        // x > 0.5 = 1
        Function<INDArray, INDArray> binarize05 =
                (arr) -> {return Transforms.step(arr.sub(0.5));};

        // create test and train datasets
        MnistLoader loaderTrain = new MnistLoader("path-to-emnist-train-images-ubyte",
                "path-to-emnist-train-labels-ubyte",
                binarize05); // preprocess func
        MnistLoader loaderTest = new MnistLoader("path-to-emnist-test-images-ubyte",
                "path-to-emnist-test-labels-ubyte",
                binarize05); // preprocess func
        // can be any batch size
        int batchSize = 64;
        // simple end condition to stop after 15 epochs
        EndCondition maxEpochs = new EpochEndCondition(15);
        // create the trainer with the options we set
        BatchedTrainer trainer = new BatchedTrainer(network, loaderTrain,
                loaderTest, batchSize, maxEpochs);

        // train the network
        trainer.train();

        // once finished, save output to zip file
        network.save(new File("model.zip"));
    }
}