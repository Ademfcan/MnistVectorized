package Data;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class MnistLoader {
    private final String TRAINIMG = "train-images.idx3-ubyte/train-images.idx3-ubyte";
    private final String TRAINLABEL = "train-labels.idx1-ubyte";
    private final String TESTIMG = "t10k-images.idx3-ubyte/t10k-images.idx3-ubyte";
    private final String TESTLABEL = "t10k-labels.idx1-ubyte";

    private INDArray[] images;
    private int[] labels;
    public MnistLoader(boolean isTrain) throws IOException {
        String img = isTrain ? TRAINIMG : TESTIMG;
        String label = isTrain ? TRAINLABEL : TESTLABEL;

        labels = UbyteReader.readIDXLabels(label);

        int[][][] rawImages = UbyteReader.readIDXImages(img);
        images = new INDArray[rawImages.length];
        for(int i = 0; i<images.length;i++){
            images[i] = Nd4j.create(rawImages[i])
                    .reshape((long) rawImages[i][0].length * rawImages[i][0].length, 1).castTo(DataType.FLOAT).div(255);
        }

        if(labels.length != images.length){
            throw new IllegalStateException("Images and labels length must match!\nLen labels: " + labels.length + " Len images: " + images.length);
        }

    }

    public INDArray getImg(int index){
        return images[index];
    }

    public INDArray getImgBatch(int index, int batchSize){
        int upper = getMax(index, batchSize);
        int size = upper-index;

        INDArray ret = Nd4j.create(images[index].rows(), size);

        for(int i = index; i < upper; i++){
            ret.putColumn(i-index, images[i].getColumn(0));
        }

        return ret;
    }

    public int getLabel(int index){
        return labels[index];
    }
    public int[] getLabelBatch(int index, int batchSize){
        int upper = getMax(index, batchSize);
        int size = upper-index;

        int[] ret = new int[size];

        for(int i = index; i<upper; i++){
            ret[i-index] = labels[i];
        }

        return ret;
    }

    private int getMax(int index, int batchSize){
        int upper = index + batchSize;
        return Math.min(upper, numEntries());
    }

    public int numEntries(){
        return images.length;
    }
}

