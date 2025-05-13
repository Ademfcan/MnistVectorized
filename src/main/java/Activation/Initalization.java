package Activation;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.factory.Nd4j;

public class Initalization {
    public static void XavierInit(INDArray weights){
        int fanAvg = (int) (weights.size(0) + weights.size(1) / 2);
        NormalDistribution dist = new NormalDistribution(0, Math.sqrt(2.0 / fanAvg));
        INDArray sampled = Nd4j.rand(weights.shape(), dist);
        weights.assign(sampled); // fill in-place
    }

    public static void HeInit(INDArray weights){
        int fanIn = (int) weights.size(1); // shape [fanOut, fanIn]
        NormalDistribution dist = new NormalDistribution(0, Math.sqrt(2.0 / fanIn));
        INDArray sampled = Nd4j.rand(weights.shape(), dist);
        weights.assign(sampled); // fill in-place
    }
}
