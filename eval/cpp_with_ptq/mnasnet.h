#pragma once

void _InvertedResidual(const qaint* x, qaint* y, const string param_path,
                       const int in_channels, const int in_height, const int in_width,
                       const int out_channels, const int out_height, const int out_width,
                       const int kernel_size, const int stride, const int expansion_factor,
                       const int act_in, int& act_out) {

    constexpr bool apply_scale = true;
    const int mid_channels = in_channels * expansion_factor;

    // Pointwise
    constexpr int l0_kernel_size = 1;
    constexpr int l0_stride = 1;
    constexpr int l0_padding = 0;
    constexpr int l0_groups = 1;
    const string l0_activation = "relu";
    int l0_act_out;
    const int l0_out_channels = mid_channels;
    const int l0_out_height = conv_out_size(in_height, l0_kernel_size, l0_stride, l0_padding);
    const int l0_out_width = conv_out_size(in_width, l0_kernel_size, l0_stride, l0_padding);
    qaint y0[l0_out_channels * l0_out_height * l0_out_width];
    Conv2d(x, y0, param_path + ".layers.0", in_channels, in_height, in_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size, l0_stride, l0_padding, l0_groups, apply_scale, l0_activation, act_in, l0_act_out);

    // Depthwise
    const int l3_kernel_size = kernel_size;
    const int l3_stride = stride;
    const int l3_padding = kernel_size / 2;
    const int l3_groups = mid_channels;
    const string l3_activation = "relu";
    int l3_act_out;
    const int l3_out_channels = mid_channels;
    const int l3_out_height = conv_out_size(l0_out_height, l3_kernel_size, l3_stride, l3_padding);
    const int l3_out_width = conv_out_size(l0_out_width, l3_kernel_size, l3_stride, l3_padding);
    qaint y3[l3_out_channels * l3_out_height * l3_out_width];
    Conv2d(y0, y3, param_path + ".layers.3", l0_out_channels, l0_out_height, l0_out_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size, l3_stride, l3_padding, l3_groups, apply_scale, l3_activation, l0_act_out, l3_act_out);

    // Linear pointwise. Note that there's no activation.
    constexpr int l6_kernel_size = 1;
    constexpr int l6_stride = 1;
    constexpr int l6_padding = 0;
    constexpr int l6_groups = 1;
    const string l6_activation = "none";
    int l6_act_out;
    const int l6_out_channels = out_channels;
    const int l6_out_height = conv_out_size(l3_out_height, l6_kernel_size, l6_stride, l6_padding);
    const int l6_out_width = conv_out_size(l3_out_width, l6_kernel_size, l6_stride, l6_padding);
    Conv2d(y3, y, param_path + ".layers.6", l3_out_channels, l3_out_height, l3_out_width, l6_out_channels, l6_out_height, l6_out_width, l6_kernel_size, l6_stride, l6_padding, l6_groups, apply_scale, l6_activation, l3_act_out, l6_act_out);

    act_out = l6_act_out;
    // if x.shape == y.shape
    if (in_channels == out_channels && stride == 1) {
        const int layer_size = out_channels * out_height * out_width;
        add_layer(x, y, layer_size, param_path, act_in, l6_act_out, act_out);
    }
}


void _stack(const qaint* x, qaint* y, const string param_path,
            const int in_channels, const int in_height, const int in_width,
            const int out_channels, const int out_height, const int out_width,
            const int kernel_size, const int stride, const int expansion_factor, const int repeats,
            const int act_in, int& act_out) {

    // First one has no skip, because feature map size changes.
    _InvertedResidual(x, y, param_path + ".0", in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, stride, expansion_factor, act_in, act_out);

    for (int i = 1; i < repeats; i++) {
        const int ir_act_out = act_out;
        qaint yi[out_channels * out_height * out_width];
        for (int idx = 0; idx < out_channels * out_height * out_width; idx++)
            yi[idx] = y[idx];
        _InvertedResidual(yi, y, param_path + "." + to_string(i), out_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, 1, expansion_factor, ir_act_out, act_out);
    }
}
