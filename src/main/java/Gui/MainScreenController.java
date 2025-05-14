package Gui;

import Data.MnistLoader;
import Network.NNetwork;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Objects;
import java.util.ResourceBundle;


public class MainScreenController implements Initializable {
    @FXML
    private Label mnistOutput;

    @FXML
    private GridPane mnistInput;

    @FXML
    private HBox container;

    @FXML
    private GridPane mainScene;

    @FXML
    private HBox header;

    @FXML
    private Button resetInput;

    @FXML
    private Button demo;

    @FXML
    private VBox sideBox;

    private Pane[][] panes;
    private float[][] data;

    private NNetwork network;
    private MnistLoader loader;



    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        initBindings();
        initGrid();

        resetInput.setOnMouseClicked(event -> {
            reset();
        });

        demo.setOnMouseClicked(event -> {
            runOptions();
        });

        try {
            network = NNetwork.fromZip(new File(Objects.
                    requireNonNull(getClass().getClassLoader().getResource("models/Out512.zip")).getPath()));

            loader = new MnistLoader("trainingdata/emnist-digits-train-images-idx3-ubyte",
                    "trainingdata/emnist-digits-train-labels-idx1-ubyte");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    private void initBindings(){
        container.prefWidthProperty().bind(mainScene.widthProperty());
        container.prefHeightProperty().bind(mainScene.heightProperty().subtract(header.heightProperty()));

        mnistInput.prefWidthProperty().bind(container.widthProperty().multiply(0.7));
        mnistInput.prefHeightProperty().bind(container.heightProperty());

        sideBox.prefWidthProperty().bind(container.widthProperty().subtract(mnistInput.widthProperty()));
        sideBox.prefHeightProperty().bind(container.heightProperty());
    }

    private void initGrid(){
        mainScene.setOnDragDetected(event -> {
            mainScene.startFullDrag();
        });

        data = new float[28][28];
        panes = new Pane[28][28];

        mnistInput.setStyle("-fx-background-color: #5f46");
        mnistInput.setHgap(2);
        mnistInput.setVgap(2);
        configureGridSize(mnistInput, 28, 28);

        for(int i = 0; i<28; i++){
            for(int j = 0; j<28; j++){
                Pane pane = new Pane();

                setupPane(i, j, pane);
                setPaneToggled(i, j, pane,false);

                mnistInput.add(pane, j, i);
                panes[i][j] = pane;
            }
        }
    }

    private void configureGridSize(GridPane grid, int numCols, int numRows) {
        grid.getColumnConstraints().clear();
        grid.getRowConstraints().clear();

        for (int i = 0; i < numCols; i++) {
            ColumnConstraints col = new ColumnConstraints();
            col.setPercentWidth(100.0 / numCols);  // Equal column widths
            grid.getColumnConstraints().add(col);
        }

        for (int i = 0; i < numRows; i++) {
            RowConstraints row = new RowConstraints();
            row.setPercentHeight(100.0 / numRows);  // Equal row heights
            grid.getRowConstraints().add(row);
        }
    }

    private void setupPane(int i, int j, Pane pane){
        pane.setOnMouseDragOver(event -> {
            setPaneToggled(i, j, pane, true);
        });

        pane.setOnMouseClicked(event -> {
            setPaneToggled(i, j, pane, true);
        });

        pane.setOnMouseDragged(event -> {
            setPaneToggled(i, j, pane, true);
        });
    }

    private void setPaneToggled(int i, int j, Pane pane, boolean toggle){
        if (toggle){
            pane.setStyle("-fx-background-color: #000");
        }
        else{
            pane.setStyle("-fx-background-color: #fff");
        }

        data[i][j] = toggle ? 1 : 0;

        if(toggle){update();}
    }

    private void runOptions(){
        new Thread(() -> {

            for (int i = 0; i < loader.numEntries(); i++) {
                final int idx = i;
                Platform.runLater(() -> {
                    INDArray image = loader.getFeatures(idx, 1).reshape(28, 28);
                    int correct = loader.getExpectedOutputs(idx, 1).getColumn(0).argMax(0).getInt(0);

                    for (int j = 0; j < 28; j++) {
                        for (int k = 0; k < 28; k++) {
                            double val = image.getDouble(j, k);
                            setPaneToggled(j, k, panes[j][k], val > 0.5);
                        }
                    }

                    mnistOutput.setText("Pred: " + getNetworkPrediction() + " Correct: "  + correct);
                });

                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }

    private void update(){
        mnistOutput.setText("Pred: " + getNetworkPrediction());
    }

    private void reset(){
        update();
        for(int i = 0; i<28; i++){
            for(int j = 0; j<28; j++){
                setPaneToggled(i, j, panes[i][j], false);
            }
        }
    }

    private INDArray getInput(){
//        return Nd4j.create(applyBinaryDilation(data)).reshape(784,1);
        return Nd4j.create(data).reshape(784,1);
    }

    private float[][] applyGaussianBlur(float[][] input) {
        float[][] output = new float[28][28];

        // 3x3 Gaussian kernel (approximation)
        float[][] kernel = {
                {1f / 8, 1f / 8, 1f / 8},
                {1f / 8, 2f / 8, 1f / 8},
                {1f / 8, 1f / 8, 1f / 8}
        };

        for (int i = 1; i < 27; i++) {
            for (int j = 1; j < 27; j++) {
                float sum = 0;
                for (int ki = -1; ki <= 1; ki++) {
                    for (int kj = -1; kj <= 1; kj++) {
                        sum += input[i + ki][j + kj] * kernel[ki + 1][kj + 1];
                    }
                }
                output[i][j] = sum;
            }
        }

        return output;
    }

    private float[][] applyBinaryDilation(float[][] input) {
        float[][] output = new float[28][28];

        for (int i = 1; i < 27; i++) {
            for (int j = 1; j < 27; j++) {
                boolean shouldSet = false;

                for (int ki = -1; ki <= 1; ki++) {
                    for (int kj = -1; kj <= 1; kj++) {
                        if (input[i + ki][j + kj] > 0.5f) {
                            shouldSet = true;
                            break;
                        }
                    }
                    if (shouldSet) break;
                }

                output[i][j] = shouldSet ? 1f : 0f;
            }
        }

        return output;
    }


    private int getNetworkPrediction(){
        network.runInference(getInput());
        return network.outputLayer.getPredictions().getInt(0);
    }

}
