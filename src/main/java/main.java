import Activation.ActivationFunction;
import Data.MnistLoader;
import Layers.InputLayer;
import Layers.Layer;
import Layers.OutputLayer;
import Learning.ConstantLearningRate;
import Learning.LearningRateProvider;
import Network.NNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.logging.Logger;

public class main {
    public static void main(String[] args) throws IOException {
        int batchSize = 64;
        int numEpochs = 50;

        LearningRateProvider constant = new ConstantLearningRate(0.015);
        NNetwork network = new NNetwork(
                new InputLayer(28 * 28),
                new OutputLayer(constant, 128, 10),
                new Layer(ActivationFunction.RELU, constant, 28 * 28, 128),
                new Layer(ActivationFunction.RELU, constant, 128, 128)
        );

        MnistLoader loaderTrain = new MnistLoader(true);
        MnistLoader loaderTest = new MnistLoader(false);

        int numTrain = loaderTrain.numEntries();
        int[] indices = new int[numTrain];
        for (int i = 0; i < numTrain; i++) {
            indices[i] = i;
        }
        Random rng = new Random();

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // Shuffle indices
            for (int i = numTrain - 1; i >= 1; i--) {
                int j = rng.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }

            // Training
            for (int batchStart = 0; batchStart < numTrain; batchStart += batchSize) {
                int actualBatchSize = Math.min(batchSize, numTrain - batchStart);

                // Gather shuffled batch
                INDArray imageInput = null;
                int[] correctLabels = new int[actualBatchSize];
                for (int b = 0; b < actualBatchSize; b++) {
                    int idx = indices[batchStart + b];
                    if (imageInput == null)
                        imageInput = loaderTrain.getImg(idx).dup();
                    else
                        imageInput = org.nd4j.linalg.factory.Nd4j.hstack(imageInput, loaderTrain.getImg(idx));
                    correctLabels[b] = loaderTrain.getLabel(idx);
                }

                network.runInference(imageInput);
                network.runBackprop(correctLabels);

                if ((batchStart / batchSize) % 100 == 0) {
                    Logger.getGlobal().info("Epoch " + (epoch + 1) + " batch " + batchStart + "/" + numTrain);
                }
            }

            // Evaluate on test set after each epoch
            int correct = 0, total = 0;
            for (int i = 0; i < loaderTest.numEntries(); i++) {
                INDArray imageInput = loaderTest.getImg(i);
                int correctLabel = loaderTest.getLabel(i);
                network.runInference(imageInput);
                int guessedIndex = network.outputLayer.getPredictions().getInt(0);
                if (guessedIndex == correctLabel) {
                    correct++;
                }
                total++;
            }
            double accuracy = correct / (double) total;
            Logger.getGlobal().info("Epoch " + (epoch + 1)
                    + " Test correct: " + correct + " / " + total + " = " + accuracy);
        }

        network.save(new File("out.zip"));
    }
}