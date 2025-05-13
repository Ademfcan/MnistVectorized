package Data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public abstract class DataLoader {
    public abstract INDArray getFeatures(int index, int batchSize);
    public abstract INDArray getFeatures(int[] indexes);
    public abstract INDArray getExpectedOutputs(int index, int batchSize);
    public abstract INDArray getExpectedOutputs(int[] indexes);
    public abstract int numEntries();

    public int getMax(int index, int batchSize){
        int upper = index + batchSize;
        return Math.min(upper, numEntries());
    }
}
