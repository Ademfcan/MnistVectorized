package Data;

public class preprocessing {
    public static float[][] applyGaussianBlur(float[][] input) {
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

    public static float[][] applyBinaryDilation(float[][] input) {
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

}
