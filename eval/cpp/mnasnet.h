#pragma once

void _InvertedResidual(const float* x, float* y, const string param_path,
                       const int in_channels, const int in_height, const int in_width,
                       const int out_channels, const int out_height, const int out_width,
                       const int kernel_size, const int stride, const int expansion_factor) {

    constexpr bool apply_bias = false;
    const int mid_channels = in_channels * expansion_factor;

    // Pointwise
    constexpr int l0_kernel_size = 1;
    constexpr int l0_stride = 1;
    constexpr int l0_padding = 0;
    constexpr int l0_groups = 1;
    const int l0_out_channels = mid_channels;
    const int l0_out_height = conv_out_size(in_height, l0_kernel_size, l0_stride, l0_padding);
    const int l0_out_width = conv_out_size(in_width, l0_kernel_size, l0_stride, l0_padding);
    float y0[l0_out_channels * l0_out_height * l0_out_width];
    Conv2d(x, y0, param_path + ".layers.0", in_channels, in_height, in_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size, l0_stride, l0_padding, l0_groups, apply_bias);

    const int l1_out_channels = mid_channels;
    const int l1_out_height = l0_out_height;
    const int l1_out_width = l0_out_width;
    BatchNorm2d(y0, param_path + ".layers.1", l1_out_channels, l1_out_height, l1_out_width);

    const int l2_out_channels = mid_channels;
    const int l2_out_height = l1_out_height;
    const int l2_out_width = l1_out_width;
    ReLU(y0, l2_out_channels, l2_out_height, l2_out_width);

    // Depthwise
    const int l3_kernel_size = kernel_size;
    const int l3_stride = stride;
    const int l3_padding = kernel_size / 2;
    const int l3_groups = mid_channels;
    const int l3_out_channels = mid_channels;
    const int l3_out_height = conv_out_size(l2_out_height, l3_kernel_size, l3_stride, l3_padding);
    const int l3_out_width = conv_out_size(l2_out_width, l3_kernel_size, l3_stride, l3_padding);
    float y3[l3_out_channels * l3_out_height * l3_out_width];
    Conv2d(y0, y3, param_path + ".layers.3", l2_out_channels, l2_out_height, l2_out_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size, l3_stride, l3_padding, l3_groups, apply_bias);

    const int l4_out_channels = mid_channels;
    const int l4_out_height = l3_out_height;
    const int l4_out_width = l3_out_width;
    BatchNorm2d(y3, param_path + ".layers.4", l4_out_channels, l4_out_height, l4_out_width);

    const int l5_out_channels = mid_channels;
    const int l5_out_height = l4_out_height;
    const int l5_out_width = l4_out_width;
    ReLU(y3, l5_out_channels, l5_out_height, l5_out_width);

    // Linear pointwise. Note that there's no activation.
    constexpr int l6_kernel_size = 1;
    constexpr int l6_stride = 1;
    constexpr int l6_padding = 0;
    constexpr int l6_groups = 1;
    const int l6_out_channels = out_channels;
    const int l6_out_height = conv_out_size(l5_out_height, l6_kernel_size, l6_stride, l6_padding);
    const int l6_out_width = conv_out_size(l5_out_width, l6_kernel_size, l6_stride, l6_padding);
    Conv2d(y3, y, param_path + ".layers.6", l5_out_channels, l5_out_height, l5_out_width, l6_out_channels, l6_out_height, l6_out_width, l6_kernel_size, l6_stride, l6_padding, l6_groups, apply_bias);

    const int l7_out_channels = out_channels;
    const int l7_out_height = l6_out_height;
    const int l7_out_width = l6_out_width;
    BatchNorm2d(y, param_path + ".layers.7", l7_out_channels, l7_out_height, l7_out_width);

    // if x.shape == y.shape
    if (in_channels == out_channels && stride == 1) {
        for (int idx = 0; idx < out_channels * out_height * out_width; idx++)
            y[idx] += x[idx];
    }
}


void _stack(const float* x, float* y, const string param_path,
            const int in_channels, const int in_height, const int in_width,
            const int out_channels, const int out_height, const int out_width,
            const int kernel_size, const int stride, const int expansion_factor, const int repeats) {

    // First one has no skip, because feature map size changes.
    _InvertedResidual(x, y, param_path + ".0", in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, stride, expansion_factor);

    for (int i = 1; i < repeats; i++) {
        float yi[out_channels * out_height * out_width];
        for (int idx = 0; idx < out_channels * out_height * out_width; idx++)
            yi[idx] = y[idx];
        _InvertedResidual(yi, y, param_path + "." + to_string(i), out_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, 1, expansion_factor);
    }
}
