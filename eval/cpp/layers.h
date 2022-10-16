#pragma once

void conv_layer(const float* x, float* y, const string param_path,
                const int in_channels, const int in_height, const int in_width,
                const int out_channels, const int out_height, const int out_width,
                const int kernel_size, const int stride, const bool apply_bn_relu) {

    const int padding = (kernel_size - 1) / 2;
    constexpr int groups = 1;
    constexpr bool apply_bias = false;
    Conv2d(x, y, param_path + ".0", in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, stride, padding, groups, apply_bias);

    if (apply_bn_relu) {
        BatchNorm2d(y, param_path + ".1", out_channels, out_height, out_width);
        ReLU(y, out_channels, out_height, out_width);
    }
}


void depth_layer_3x3(const float* x, float* y, const string param_path,
                     const int in_channels, const int height, const int width) {

    constexpr int out_channels = 1;
    constexpr int kernel_size = 3;
    constexpr int stride = 1;
    constexpr int padding = (kernel_size - 1) / 2;
    constexpr int groups = 1;
    constexpr bool apply_bias = true;
    Conv2d(x, y, param_path + ".0", in_channels, height, width, out_channels, height, width, kernel_size, stride, padding, groups, apply_bias);
    Sigmoid(y, out_channels, height, width);
}
