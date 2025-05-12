import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class main {
    public static void main(String[] args) {
        // Create 2x3 matrix A
        INDArray A = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });

        // Create 3x2 matrix B
        INDArray B = Nd4j.create(new double[][] {
                {7, 8},
                {9, 10},
                {11, 12}
        });

        // Multiply A * B -> should result in a 2x2 matrix
        INDArray C = A.mmul(B);

        // Print the result
        System.out.println("Result of A x B:");
        System.out.println(C);
    }
}
