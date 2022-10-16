#pragma once

void ReLU(float* x, const int channels, const int height, const int width) {
    for (int idx = 0; idx < channels * height * width; idx++)
        x[idx] = max(0.0f, x[idx]);
}


void celu(float* x, const int channels, const int height, const int width) {
    for (int idx = 0; idx < channels * height * width; idx++)
        x[idx] = max(0.0f, x[idx]) + min(0.0f, exp(x[idx]) - 1);
}


void Sigmoid(float* x, const int channels, const int height, const int width) {
    for (int idx = 0; idx < channels * height * width; idx++)
        x[idx] = 1.0 / (1.0 + exp(-x[idx]));
}
