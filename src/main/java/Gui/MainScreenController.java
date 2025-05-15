package Gui;

import Data.MnistLoader;
import Data.preprocessing;
import Network.NNetwork;
import Network.ZipHelper;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
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
    private Button checkInput;

    @FXML
    private Button playDemo;

    @FXML
    private VBox sideBox;

    @FXML
    private Button loadModel;

    @FXML
    private Label headerLabel;

    @FXML
    private Slider brushSize;

    private Pane[][] panes;
    private float[][] data;

    private NNetwork network;
    private MnistLoader loader;
    private Thread demoThread;



    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        initBindings();
        initGrid();

        resetInput.setOnMouseClicked(event -> {
            reset();
        });

        playDemo.setOnMouseClicked(event -> {
            playDemo.setText(toggleDemo() ? "Stop" : "Demo");
        });

        checkInput.setOnMouseClicked(event -> {
            updatePrediction(true);
        });

        loadModel.setOnAction(event -> {
            loadNewModel();
        });

        initNNetwork();
    }

    /**
    * Load a nnetwork file from a saved zip, and also load a test dataset
    */
    private void initNNetwork(){
        try {
            network = ZipHelper.fromZip(
                    Objects.requireNonNull(getClass().getClassLoader().getResourceAsStream("models/out.zip")));

            loader = new MnistLoader(true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Initialize dynamic layout
     */
    private void initBindings(){
        container.prefWidthProperty().bind(mainScene.widthProperty());
        container.prefHeightProperty().bind(mainScene.heightProperty().subtract(header.heightProperty()));

        mnistInput.prefWidthProperty().bind(container.widthProperty().multiply(0.7));
        mnistInput.prefHeightProperty().bind(container.heightProperty());

        sideBox.prefWidthProperty().bind(container.widthProperty().subtract(mnistInput.widthProperty()));
        sideBox.prefHeightProperty().bind(container.heightProperty());

        header.spacingProperty().bind(header.widthProperty().divide(2).
                subtract(loadModel.widthProperty()).
                subtract(headerLabel.widthProperty().divide(2)).
                subtract(5)); // 5px left margin

        brushSize.prefWidthProperty().bind(sideBox.widthProperty().multiply(0.65));



    }

    /**
     * Initialize the input grid, setting it to a be 28x28 matrix that can be drawn on
     * Also initialize the panes that will recieve the draw input, and change color when drawn on
     */
    private void initGrid(){
        mainScene.setOnDragDetected(event -> {
            mainScene.startFullDrag();
        });

        data = new float[28][28];
        panes = new Pane[28][28];

        mnistInput.setHgap(1);
        mnistInput.setVgap(1);
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

    /**
     * Listen to click, drag and drag start events
     */
    private void setupPane(int i, int j, Pane pane){
        pane.setOnMouseDragOver(event -> {
            setPaneToggled(i, j, true, (int) brushSize.getValue());
        });

        pane.setOnMouseClicked(event -> {
            setPaneToggled(i, j, true, (int) brushSize.getValue());
        });

        pane.setOnMouseDragged(event -> {
            setPaneToggled(i, j, true, (int) brushSize.getValue());
        });
    }

    /**
     * Same as setPaneToggled below, but with a bigger brush size
     */
    private void setPaneToggled(int i, int j, boolean toggle, int brushSize) {
        int radius = brushSize / 2;
        for (int di = -radius; di <= radius; di++) {
            for (int dj = -radius; dj <= radius; dj++) {
                int ni = i + di;
                int nj = j + dj;
                if (ni >= 0 && ni < 28 && nj >= 0 && nj < 28) {
                    setPaneToggled(ni, nj, panes[ni][nj], toggle);
                }
            }
        }
    }

    /**
     * Set the color of a pane to be black when toggled and white otherwise
     * Additionaly, set the underlying data array to be toggled aswell
     */
    private void setPaneToggled(int i, int j, Pane pane, boolean toggle){
        if (toggle){
            // toggle black
            pane.setStyle("-fx-background-color: #2A2B2E");
        }
        else{
            // toggle white
            pane.setStyle("-fx-background-color: #F7F9F9");
        }

        data[i][j] = toggle ? 1 : 0;
    }

    /**
     * Toggle a demo by displaying the loaded dataset on screen
     * Starts a background demo thread, and displays a new image and its prediction every second
     *
     * @return true if a demo started, and false if it stopped a running demo
     */
    private boolean toggleDemo(){
        if(demoThread == null){
            demoThread = new Thread(() -> {
                for (int i = 0; i < loader.numEntries() && !Thread.currentThread().isInterrupted(); i++) {
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

                        mnistOutput.setText("P: " + getNetworkPrediction(false) + " A: "  + correct);
                    });

                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException e) {
                        // Exit loop if interrupted
                        break;
                    }
                }
            });

            demoThread.start();
            return true;
        }
        else {
            demoThread.interrupt();
            demoThread = null;
            reset();
            mnistOutput.setText("P: ");

            return false;
        }
    }

    public void stopDemoThread() {
        if (demoThread != null && demoThread.isAlive()) {
            demoThread.interrupt();
            demoThread = null;
        }
    }

    private void reset(){
        updatePrediction(true);
        for(int i = 0; i<28; i++){
            for(int j = 0; j<28; j++){
                setPaneToggled(i, j, panes[i][j], false);
            }
        }
    }

    private void updatePrediction(boolean preprocess){
        mnistOutput.setText("P: " + getNetworkPrediction(false));
    }

    private INDArray getInput(boolean preprocess){
        if (preprocess){
            return Nd4j.create(preprocessing.applyBinaryDilation(data)).reshape(784,1);
        }
        else{
            return Nd4j.create(data).reshape(784,1);
        }
    }

    private int getNetworkPrediction(boolean preprocess){
        network.runInference(getInput(preprocess));
        return network.outputLayer.getPredictions().getInt(0);
    }

    /**
     * Load a new model from a zip file
     */
    private void loadNewModel() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Open Model Zip File");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("ZIP files", "*.zip"));
        File selectedFile = fileChooser.showOpenDialog(mnistInput.getScene().getWindow());

        if (selectedFile != null) {
            try {
                stopDemoThread(); // stop demo if running
                network = ZipHelper.fromZip(selectedFile);
                mnistOutput.setText("New model loaded successfully.");
            } catch (IOException e) {
                mnistOutput.setText("Failed to load model: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }


}
