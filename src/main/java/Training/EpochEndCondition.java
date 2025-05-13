package Training;

import Network.NNetwork;

import java.util.List;

public class EpochEndCondition extends EndCondition {
    int maxEpochs;
    public EpochEndCondition(int maxEpochs) {
        this.maxEpochs = maxEpochs;
    }

    @Override
    boolean isFinished(List<Double> accuracyOverTime, int numEpochs, NNetwork network) {
        return numEpochs >= maxEpochs;
    }
}
