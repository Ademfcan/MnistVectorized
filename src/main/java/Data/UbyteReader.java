package Data;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

public class UbyteReader {
    public static int[][][] readIDXImages(String filename) throws IOException {
        try (DataInputStream dis = new DataInputStream(UbyteReader.class.getClassLoader().getResourceAsStream(filename))) {
            dis.readInt(); // magic number
            int numImages = dis.readInt();
            int numRows = dis.readInt();
            int numCols = dis.readInt();

            int[][][] images = new int[numImages][numRows][numCols];

            for (int i = 0; i < numImages; i++) {
                for (int r = 0; r < numRows; r++) {
                    for (int c = 0; c < numCols; c++) {
                        images[i][r][c] = dis.readUnsignedByte();  // 0–255
                    }
                }
            }

            return images;
        }
    }

    public static int[] readIDXLabels(String filename) throws IOException {
        try (DataInputStream dis = new DataInputStream(UbyteReader.class.getClassLoader().getResourceAsStream(filename))) {
            dis.readInt(); // magic number
            int numItems = dis.readInt();
            int[] labels = new int[numItems];

            for (int i = 0; i < numItems; i++) {
                labels[i] = dis.readUnsignedByte();  // 0–9 for MNIST labels
            }

            return labels;
        }
    }

    public static void main(String[] args) throws IOException {
        int[][][] images = readIDXImages("train-images.idx3-ubyte/train-images.idx3-ubyte");
        int[] labels = readIDXLabels("train-labels.idx1-ubyte");

        System.out.println("First label: " + labels[0]);
        System.out.println("First image: " + Arrays.deepToString(images[0]));
    }
}
