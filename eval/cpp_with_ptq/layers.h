#pragma once

void conv_layer(const qaint* x, qaint* y, const string param_path,
                const int in_channels, const int in_height, const int in_width,
                const int out_channels, const int out_height, const int out_width,
                const int kernel_size, const int stride,
                const int act_in, int& act_out) {

    const int padding = (kernel_size - 1) / 2;
    constexpr int groups = 1;
    constexpr bool apply_scale = true;
    const string activation = "relu";
    Conv2d(x, y, param_path + ".0", in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, stride, padding, groups, apply_scale, activation, act_in, act_out);
}


void depth_layer_3x3(const qaint* x, qaint* y, const string param_path,
                     const int in_channels, const int height, const int width,
                     const int act_in, int& act_out) {

    constexpr int out_channels = 1;
    constexpr int kernel_size = 3;
    constexpr int stride = 1;
    constexpr int padding = (kernel_size - 1) / 2;
    constexpr int groups = 1;
    constexpr bool apply_scale = false;
    const string activation = "sigmoid";
    Conv2d(x, y, param_path + ".0", in_channels, height, width, out_channels, height, width, kernel_size, stride, padding, groups, apply_scale, activation, act_in, act_out);
}
