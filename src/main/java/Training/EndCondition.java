package Training;

import Network.NNetwork;

import java.util.List;

public abstract class EndCondition {
    abstract boolean isFinished(List<Double> accuracyOverTime, int numEpochs, NNetwork network);
}
