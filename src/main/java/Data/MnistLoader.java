package Data;

import javafx.scene.chart.CategoryAxis;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.function.Function;

public class MnistLoader extends DataLoader{
    private static final String TRAINIMG = "trainingdata/train-images.idx3-ubyte/train-images.idx3-ubyte";
    private static final String TRAINLABEL = "trainingdata/train-labels.idx1-ubyte";
    private static final String TESTIMG = "trainingdata/t10k-images.idx3-ubyte/t10k-images.idx3-ubyte";
    private static final String TESTLABEL = "trainingdata/t10k-labels.idx1-ubyte";

    private INDArray[] images;
    private int[] labels;
    public MnistLoader(boolean isTrain) throws IOException {
        this(isTrain ? TRAINIMG : TESTIMG, isTrain ? TRAINLABEL : TESTLABEL);
    }
    public MnistLoader(boolean isTrain, Function<INDArray,INDArray> preprocess) throws IOException {
        this(isTrain ? TRAINIMG : TESTIMG, isTrain ? TRAINLABEL : TESTLABEL, preprocess);
    }

    public MnistLoader(String img, String label) throws IOException {
        this(img, label, null);
    }

    public MnistLoader(String img, String label, Function<INDArray,INDArray> preprocess) throws IOException {
        labels = UbyteReader.readIDXLabels(label);

        int[][][] rawImages = UbyteReader.readIDXImages(img);
        images = new INDArray[rawImages.length];
        for(int i = 0; i<images.length;i++){
            images[i] = Nd4j.create(rawImages[i])
                    .reshape(784, 1).castTo(DataType.FLOAT).div(255);

            if(preprocess != null){
                images[i] = preprocess.apply(images[i]);
            }
        }

        if(labels.length != images.length){
            throw new IllegalStateException("Images and labels length must match!\nLen labels: " + labels.length + " Len images: " + images.length);
        }

    }



    @Override
    public INDArray getFeatures(int index, int batchSize) {
        int upper = getMax(index, batchSize);
        int size = upper-index;

        INDArray ret = Nd4j.create(images[index].rows(), size);

        for(int i = index; i < upper; i++){
            ret.putColumn(i-index, images[i].getColumn(0));
        }

        return ret;
    }

    @Override
    public INDArray getFeatures(int[] indexes) {
        if(indexes.length < 1){
            throw new IllegalStateException("Indexes array must contain at least one element!");
        }

        INDArray ret = Nd4j.create(images[indexes[0]].rows(), indexes.length);
        for(int i = 0; i < indexes.length; i++){
            int idx = indexes[i];
            ret.putColumn(i, images[idx].getColumn(0));
        }

        return ret;
    }

    @Override
    public INDArray getExpectedOutputs(int index, int batchSize) {
        int upper = getMax(index, batchSize);
        int size = upper-index;

        INDArray oneHot = Nd4j.zeros(10, size);
        for(int i = index; i<upper;i++){
            oneHot.putScalar(labels[i], i-index, 1.0f);
        }

        return oneHot;
    }

    @Override
    public INDArray getExpectedOutputs(int[] indexes) {
        if(indexes.length < 1){
            throw new IllegalStateException("Indexes array must contain at least one element!");
        }
        INDArray oneHot = Nd4j.create(10, indexes.length);
        for(int i = 0; i < indexes.length; i++){
            int idx = indexes[i];
            oneHot.putScalar(labels[idx], i, 1);
        }

        return oneHot;
    }

    public int numEntries(){
        return images.length;
    }
}

