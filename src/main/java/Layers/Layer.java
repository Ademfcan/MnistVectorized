package Layers;
import Activation.ActivationFunction;
import Learning.LearningRateProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.logging.Logger;

public class Layer {
    protected INDArray weights;
    protected INDArray biases;

    protected INDArray activations;
    protected INDArray errorSignal;
    protected INDArray v;

    protected INDArray dLdWs;
    protected INDArray dLdBs;

    protected Layer previous;
    protected Layer next;

    protected final ActivationFunction activationFunc;
    protected final LearningRateProvider learningRateProvider;

    private final int shapeIn;
    private final int shapeOut;

    public Layer(ActivationFunction activationFunc, LearningRateProvider learningRateProvider, int shapeIn, int shapeOut){
        this.activationFunc = activationFunc;
        this.learningRateProvider = learningRateProvider;
        this.shapeIn = shapeIn;
        this.shapeOut = shapeOut;

        createWeights(shapeIn, shapeOut);
        this.dLdWs = Nd4j.zerosLike(weights);
        this.dLdBs = Nd4j.zerosLike(biases);

    }

    private void createWeights(int shapeIn, int shapeOut){
        this.weights = Nd4j.create(shapeOut, shapeIn);
        this.biases = Nd4j.zeros(shapeOut, 1);

        activationFunc.initializeWeights(weights);
    }

    public void setPrevious(Layer previous) {
        this.previous = previous;
    }

    public void setNext(Layer next) {
        this.next = next;
    }


    public void forward(INDArray previousActivations){
        v = weights.mmul(previousActivations).addColumnVector(biases);
        activations = activationFunc.func(v);

        if(next != null){
            next.forward(activations);
        }
//        else{
//            Logger.getGlobal().info("Layer has reached the end of forward propagation, next is null");
//        }
    }


    public void backward(INDArray nextErrorSignal, INDArray nextWeights){
        errorSignal = nextWeights.transpose().mmul(nextErrorSignal)
                .mul(activationFunc.derivative(v));
        int batchSize = errorSignal.columns();

        INDArray dLdW = errorSignal.mmul(previous.activations.transpose()).div(batchSize);
        INDArray dLdB = errorSignal.sum(1).reshape(errorSignal.rows(), 1).div(batchSize);

        double learningRate = learningRateProvider.getLearningRate(this);

        weights.subi(dLdW.mul(learningRate));
        biases.subi(dLdB.mul(learningRate));

        if(previous != null){
            previous.backward(errorSignal, weights);
        }
//        else{
//            Logger.getGlobal().info("Layer has reached the end of backpropagation, previous is null");
//        }
    }

    public INDArray getWeights() {
        return weights;
    }

    public INDArray getBiases() {
        return biases;
    }

    public INDArray getActivations() {
        return activations;
    }

    public INDArray getPredictions() {
        INDArray ret = Nd4j.zeros(activations.columns());
        for(int i = 0; i< activations.columns(); i++){
            ret.putScalar(i, activations.getColumn(i).argMax(0).getInt(0));
        }

        return ret;
    }

    public INDArray getErrorSignal() {
        return errorSignal;
    }

    public Layer getPrevious() {
        return previous;
    }

    public Layer getNext() {
        return next;
    }

    public int getShapeOut() {
        return shapeOut;
    }

    public int getShapeIn() {
        return shapeIn;
    }



}
