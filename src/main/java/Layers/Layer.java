package Layers;
import Activation.ActivationFunction;
import Learning.LearningRateProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Layer {
    protected INDArray weights;
    protected INDArray biases;

    protected INDArray activations;
    protected INDArray errorSignal;

    protected Layer previous;
    protected Layer next;

    private final ActivationFunction activationFunc;
    private final LearningRateProvider learningRateProvider;

    private final int weightR;
    private final int weightC;

    public Layer(ActivationFunction activationFunc, LearningRateProvider learningRateProvider, int weightR, int weightC){
        this.activationFunc = activationFunc;
        this.learningRateProvider = learningRateProvider;
        this.weightR = weightR;
        this.weightC = weightC;

        createWeights(weightR, weightC);
    }

    private void createWeights(int weightR, int weightC){
        this.weights = Nd4j.create(weightR, weightC);
        this.biases = Nd4j.create(weightR, 1);

        activationFunc.initializeWeights(weights);
        activationFunc.initializeWeights(biases);
    }

    public void setPrevious(Layer previous) {
        this.previous = previous;
    }

    public void setNext(Layer next) {
        this.next = next;
    }


    public INDArray forward(){
        INDArray v = weights.mmul(previous.activations).addColumnVector(biases);
        activations = activationFunc.func(v);

        return activations;
    }


    public INDArray backward(){
        errorSignal = next.weights.transpose().mmul(next.errorSignal)
                .mul(activationFunc.derivative(activations));

        INDArray dLdW = errorSignal.mmul(previous.activations.transpose());
        INDArray dLdB = errorSignal;

        double learningRate = learningRateProvider.getLearningRate(this);

        weights.subi(dLdW.mul(learningRate));
        biases.subi(dLdB.mul(learningRate));

        return errorSignal;
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

    public INDArray getErrorSignal() {
        return errorSignal;
    }

    public Layer getPrevious() {
        return previous;
    }

    public Layer getNext() {
        return next;
    }

    public int getWeightR() {
        return weightR;
    }

    public int getWeightC() {
        return weightC;
    }



}
