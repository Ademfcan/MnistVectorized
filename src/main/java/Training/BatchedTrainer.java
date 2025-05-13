package Training;

import Data.DataLoader;
import Network.NNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

public class BatchedTrainer {
    private final NNetwork network;
    private final int batchSize;
    private final EndCondition endCondition;
    private final DataLoader dataLoaderTrain;
    private final DataLoader dataLoaderTest;
    private final Random rng = new Random();


    public BatchedTrainer(NNetwork network, DataLoader dataLoaderTrain, DataLoader dataLoaderTest,
                                int batchSize, EndCondition endCondition) {
        this.network = network;
        this.batchSize = batchSize;
        this.endCondition = endCondition;
        this.dataLoaderTrain = dataLoaderTrain;
        this.dataLoaderTest = dataLoaderTest;
    }

    public void train(){
        int numTrain = dataLoaderTrain.numEntries();

        int[] indices = new int[numTrain];
        for (int i = 0; i < numTrain; i++) {
            indices[i] = i;
        }

        List<Double> accuracyOverTime = new ArrayList<>();
        int epoch = 0;
        while (!endCondition.isFinished(accuracyOverTime, epoch, network)) {
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
                int[] indexes = Arrays.copyOfRange(indices, batchStart, batchStart + actualBatchSize);
                INDArray featureInput = dataLoaderTrain.getFeatures(indexes);
                INDArray correctOutputs = dataLoaderTrain.getExpectedOutputs(indexes);

                network.runInference(featureInput);
                network.runBackprop(correctOutputs);

                if ((batchStart / batchSize) % 100 == 0) {
                    Logger.getGlobal().info("Epoch " + (epoch + 1) + " batch " + batchStart + "/" + numTrain);
                }
            }

            // Evaluate on test set after each epoch
            int correct = 0, total = 0;
            for (int i = 0; i < dataLoaderTest.numEntries(); i++) {
                INDArray imageInput = dataLoaderTest.getFeatures(i, 1);
                int expectedLabel = dataLoaderTest.getExpectedOutputs(i, 1)
                        .getColumn(0).argMax(0).getInt(0);
                network.runInference(imageInput);

                int guessedIndex = network.outputLayer.getPredictions().getInt(0);
                if (guessedIndex == expectedLabel) {
                    correct++;
                }
                total++;
            }
            double accuracy = correct / (double) total;
            Logger.getGlobal().info("Epoch " + (epoch + 1)
                    + " Test correct: " + correct + " / " + total + " = " + accuracy);

            epoch++;
            accuracyOverTime.add(accuracy);
        }

    }

}
