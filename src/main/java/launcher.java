

import Gui.MainScreenController;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class launcher extends Application {
    @Override
    public void start(Stage primaryStage) throws Exception {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("layout/main.fxml"));
        Parent root = loader.load();

        // Stop thread on window close
        MainScreenController controller = loader.getController();
        primaryStage.setOnCloseRequest(event -> controller.stopDemoThread());

        primaryStage.setTitle("Mnist Detector");
        primaryStage.setScene(new Scene(root, 800, 600));
        primaryStage.show();
    }
}
