#pragma once

void StandardLayer(const float* x, float* y, const string param_path,
                   const int channels, const int height, const int width,
                   const int kernel_size, const bool apply_bn_relu) {

    constexpr int stride = 1;
    constexpr bool l0_apply_bn_relu = true;
    float y0[channels * height * width];
    conv_layer(x, y0, param_path + ".conv1", channels, height, width, channels, height, width, kernel_size, stride, l0_apply_bn_relu);
    conv_layer(y0, y, param_path + ".conv2", channels, height, width, channels, height, width, kernel_size, stride, apply_bn_relu);
}


void DownconvolutionLayer(const float* x, float* y, const string param_path,
                          const int in_channels, const int in_height, const int in_width,
                          const int out_channels, const int out_height, const int out_width,
                          const int kernel_size) {

    constexpr int stride = 2;
    constexpr bool apply_bn_relu = true;
    conv_layer(x, y, param_path + ".down_conv", in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, stride, apply_bn_relu);
}


void UpconvolutionLayer(const float* x, float* y, const string param_path,
                        const int in_channels, const int in_height, const int in_width,
                        const int out_channels, const int kernel_size) {

    const int out_height = in_height * 2;
    const int out_width = in_width * 2;
    float up_x[in_channels * out_height * out_width];
    interpolate(x, up_x, "bilinear", in_channels, in_height, in_width, out_height, out_width);

    constexpr int stride = 1;
    constexpr bool apply_bn_relu = true;
    conv_layer(up_x, y, param_path + ".conv", in_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, apply_bn_relu);
}


void EncoderBlock(const float* x, float* y, const string param_path,
                  const int in_channels, const int in_height, const int in_width,
                  const int out_channels, const int out_height, const int out_width,
                  const int kernel_size) {

    float y0[out_channels * out_height * out_width];
    DownconvolutionLayer(x, y0, param_path + ".down_convolution", in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size);

    constexpr bool apply_bn_relu = true;
    StandardLayer(y0, y, param_path + ".standard_convolution", out_channels, out_height, out_width, kernel_size, apply_bn_relu);
}


void DecoderBlock(const float* x, const float* skip, const float* depth, float* y, const string param_path,
                  const int in_channels, const int in_height, const int in_width,
                  const int kernel_size, const bool apply_bn_relu, const bool plus_one) {

    const int out_height = in_height * 2;
    const int out_width = in_width * 2;
    const int out_channels = in_channels / 2;

    float y0[out_channels * out_height * out_width];
    UpconvolutionLayer(x, y0, param_path + ".up_convolution", in_channels, in_height, in_width, out_channels, kernel_size);

    // Aggregate skip and upsampled input
    const int l1_in_channels = plus_one ? in_channels + 1 : in_channels;
    float x1[l1_in_channels * out_height * out_width];
    for (int idx = 0; idx < out_channels * out_height * out_width; idx++)
        x1[idx] = y0[idx];
    for (int idx = 0; idx < out_channels * out_height * out_width; idx++)
        x1[idx + (out_channels * out_height * out_width)] = skip[idx];
    if (plus_one) {
        interpolate(depth, x1 + (in_channels * out_height * out_width), "bilinear", 1, in_height, in_width, out_height, out_width);
    }

    constexpr int stride = 1;

    constexpr bool l1_apply_bn_relu = true;
    float y1[out_channels * out_height * out_width];
    conv_layer(x1, y1, param_path + ".convolution1", l1_in_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, l1_apply_bn_relu);

    // Learn from aggregation
    const bool l2_apply_bn_relu = apply_bn_relu;
    conv_layer(y1, y, param_path + ".convolution2", out_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, l2_apply_bn_relu);
}


void FeatureExtractor(const float x[3 * test_image_height * test_image_width],
                      float layer1[channels_1 * height_2 * width_2],
                      float layer2[channels_2 * height_4 * width_4],
                      float layer3[channels_3 * height_8 * width_8],
                      float layer4[channels_4 * height_16 * width_16],
                      float layer5[channels_5 * height_32 * width_32]) {

    constexpr int depths[8] = {32, channels_1, channels_2, channels_3, 80, channels_4, 192, channels_5};
    constexpr bool apply_bias = false;

    // First layer: regular conv.
    float x0[3 * test_image_height * test_image_width];
    for (int idx = 0; idx < 3 * test_image_height * test_image_width; idx++)
        x0[idx] = x[idx];

    constexpr int l0_kernel_size = 3;
    constexpr int l0_stride = 2;
    constexpr int l0_padding = 1;
    constexpr int l0_groups = 1;
    constexpr int l0_out_channels = depths[0];
    constexpr int l0_out_height = conv_out_size(test_image_height, l0_kernel_size, l0_stride, l0_padding);
    constexpr int l0_out_width = conv_out_size(test_image_width, l0_kernel_size, l0_stride, l0_padding);
    float y0[l0_out_channels * l0_out_height * l0_out_width];
    Conv2d(x0, y0, "layer1.0", 3, test_image_height, test_image_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size, l0_stride, l0_padding, l0_groups, apply_bias);

    constexpr int l1_out_channels = depths[0];
    constexpr int l1_out_height = l0_out_height;
    constexpr int l1_out_width = l0_out_width;
    BatchNorm2d(y0, "layer1.1", l1_out_channels, l1_out_height, l1_out_width);

    constexpr int l2_out_channels = depths[0];
    constexpr int l2_out_height = l1_out_height;
    constexpr int l2_out_width = l1_out_width;
    ReLU(y0, l2_out_channels, l2_out_height, l2_out_width);

    // Depthwise separable, no skip.
    constexpr int l3_kernel_size = 3;
    constexpr int l3_stride = 1;
    constexpr int l3_padding = 1;
    constexpr int l3_groups = depths[0];
    constexpr int l3_out_channels = depths[0];
    constexpr int l3_out_height = conv_out_size(l2_out_height, l3_kernel_size, l3_stride, l3_padding);
    constexpr int l3_out_width = conv_out_size(l2_out_width, l3_kernel_size, l3_stride, l3_padding);
    float y3[l3_out_channels * l3_out_height * l3_out_width];
    Conv2d(y0, y3, "layer1.3", l2_out_channels, l2_out_height, l2_out_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size, l3_stride, l3_padding, l3_groups, apply_bias);

    constexpr int l4_out_channels = depths[0];
    constexpr int l4_out_height = l3_out_height;
    constexpr int l4_out_width = l3_out_width;
    BatchNorm2d(y3, "layer1.4", l4_out_channels, l4_out_height, l4_out_width);

    constexpr int l5_out_channels = depths[0];
    constexpr int l5_out_height = l4_out_height;
    constexpr int l5_out_width = l4_out_width;
    ReLU(y3, l5_out_channels, l5_out_height, l5_out_width);

    constexpr int l6_kernel_size = 1;
    constexpr int l6_stride = 1;
    constexpr int l6_padding = 0;
    constexpr int l6_groups = 1;
    constexpr int l6_out_channels = depths[1];
    constexpr int l6_out_height = conv_out_size(l5_out_height, l6_kernel_size, l6_stride, l6_padding);
    constexpr int l6_out_width = conv_out_size(l5_out_width, l6_kernel_size, l6_stride, l6_padding);
    Conv2d(y3, layer1, "layer1.6", l5_out_channels, l5_out_height, l5_out_width, l6_out_channels, l6_out_height, l6_out_width, l6_kernel_size, l6_stride, l6_padding, l6_groups, apply_bias);

    constexpr int l7_out_channels = depths[1];
    constexpr int l7_out_height = l6_out_height;
    constexpr int l7_out_width = l6_out_width;
    BatchNorm2d(layer1, "layer1.7", l7_out_channels, l7_out_height, l7_out_width);

    // MNASNet blocks: stacks of inverted residuals.
    constexpr int l8_kernel_size = 3;
    constexpr int l8_stride = 2;
    constexpr int l8_expansion_factor = 3;
    constexpr int l8_repeats = 3;
    constexpr int l8_out_channels = depths[2];
    constexpr int l8_out_height = stack_out_size(l7_out_height, l8_kernel_size, l8_stride);
    constexpr int l8_out_width = stack_out_size(l7_out_width, l8_kernel_size, l8_stride);
    _stack(layer1, layer2, "layer2.0", l7_out_channels, l7_out_height, l7_out_width, l8_out_channels, l8_out_height, l8_out_width, l8_kernel_size, l8_stride, l8_expansion_factor, l8_repeats);

    constexpr int l9_kernel_size = 5;
    constexpr int l9_stride = 2;
    constexpr int l9_expansion_factor = 3;
    constexpr int l9_repeats = 3;
    constexpr int l9_out_channels = depths[3];
    constexpr int l9_out_height = stack_out_size(l8_out_height, l9_kernel_size, l9_stride);
    constexpr int l9_out_width = stack_out_size(l8_out_width, l9_kernel_size, l9_stride);
    _stack(layer2, layer3, "layer3.0", l8_out_channels, l8_out_height, l8_out_width, l9_out_channels, l9_out_height, l9_out_width, l9_kernel_size, l9_stride, l9_expansion_factor, l9_repeats);

    constexpr int l10_kernel_size = 5;
    constexpr int l10_stride = 2;
    constexpr int l10_expansion_factor = 6;
    constexpr int l10_repeats = 3;
    constexpr int l10_out_channels = depths[4];
    constexpr int l10_out_height = stack_out_size(l9_out_height, l10_kernel_size, l10_stride);
    constexpr int l10_out_width = stack_out_size(l9_out_width, l10_kernel_size, l10_stride);
    float y10[l10_out_channels * l10_out_height * l10_out_width];
    _stack(layer3, y10, "layer4.0", l9_out_channels, l9_out_height, l9_out_width, l10_out_channels, l10_out_height, l10_out_width, l10_kernel_size, l10_stride, l10_expansion_factor, l10_repeats);

    constexpr int l11_kernel_size = 3;
    constexpr int l11_stride = 1;
    constexpr int l11_expansion_factor = 6;
    constexpr int l11_repeats = 2;
    constexpr int l11_out_channels = depths[5];
    constexpr int l11_out_height = stack_out_size(l10_out_height, l11_kernel_size, l11_stride);
    constexpr int l11_out_width = stack_out_size(l10_out_width, l11_kernel_size, l11_stride);
    _stack(y10, layer4, "layer4.1", l10_out_channels, l10_out_height, l10_out_width, l11_out_channels, l11_out_height, l11_out_width, l11_kernel_size, l11_stride, l11_expansion_factor, l11_repeats);

    constexpr int l12_kernel_size = 5;
    constexpr int l12_stride = 2;
    constexpr int l12_expansion_factor = 6;
    constexpr int l12_repeats = 4;
    constexpr int l12_out_channels = depths[6];
    constexpr int l12_out_height = stack_out_size(l11_out_height, l12_kernel_size, l12_stride);
    constexpr int l12_out_width = stack_out_size(l11_out_width, l12_kernel_size, l12_stride);
    float y12[l12_out_channels * l12_out_height * l12_out_width];
    _stack(layer4, y12, "layer5.0", l11_out_channels, l11_out_height, l11_out_width, l12_out_channels, l12_out_height, l12_out_width, l12_kernel_size, l12_stride, l12_expansion_factor, l12_repeats);

    constexpr int l13_kernel_size = 3;
    constexpr int l13_stride = 1;
    constexpr int l13_expansion_factor = 6;
    constexpr int l13_repeats = 1;
    constexpr int l13_out_channels = depths[7];
    constexpr int l13_out_height = stack_out_size(l12_out_height, l13_kernel_size, l13_stride);
    constexpr int l13_out_width = stack_out_size(l12_out_width, l13_kernel_size, l13_stride);
    _stack(y12, layer5, "layer5.1", l12_out_channels, l12_out_height, l12_out_width, l13_out_channels, l13_out_height, l13_out_width, l13_kernel_size, l13_stride, l13_expansion_factor, l13_repeats);

}


void FeatureShrinker(const float layer1[channels_1 * height_2 * width_2],
                     const float layer2[channels_2 * height_4 * width_4],
                     const float layer3[channels_3 * height_8 * width_8],
                     const float layer4[channels_4 * height_16 * width_16],
                     const float layer5[channels_5 * height_32 * width_32],
                     float features_half[fpn_output_channels * height_2 * width_2],
                     float features_quarter[fpn_output_channels * height_4 * width_4],
                     float features_one_eight[fpn_output_channels * height_8 * width_8],
                     float features_one_sixteen[fpn_output_channels * height_16 * width_16]) {

    // Module that adds a FPN from on top of a set of feature maps. This is based on
    // `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    // The feature maps are currently supposed to be in increasing depth order.
    // The input to the model is expected to be an OrderedDict[Tensor], containing
    // the feature maps on top of which the FPN will be added.

    constexpr int stride = 1;
    constexpr int groups = 1;
    constexpr bool apply_bias = true;

    constexpr int inner_kernel_size = 1;
    constexpr int inner_padding = 0;
    constexpr int layer_kernel_size = 3;
    constexpr int layer_padding = 1;

    // layer5
    float inner5[fpn_output_channels * height_32 * width_32];
    Conv2d(layer5, inner5, "fpn.inner_blocks.4", channels_5, height_32, width_32, fpn_output_channels, height_32, width_32, inner_kernel_size, stride, inner_padding, groups, apply_bias);


    // layer4
    float inner4[fpn_output_channels * height_16 * width_16];
    Conv2d(layer4, inner4, "fpn.inner_blocks.3", channels_4, height_16, width_16, fpn_output_channels, height_16, width_16, inner_kernel_size, stride, inner_padding, groups, apply_bias);

    float top_down4[fpn_output_channels * height_16 * width_16];
    interpolate(inner5, top_down4, "nearest", fpn_output_channels, height_32, width_32, height_16, width_16);
    for (int idx = 0; idx < fpn_output_channels * height_16 * width_16; idx++)
        inner4[idx] += top_down4[idx];

    Conv2d(inner4, features_one_sixteen, "fpn.layer_blocks.3", fpn_output_channels, height_16, width_16, fpn_output_channels, height_16, width_16, layer_kernel_size, stride, layer_padding, groups, apply_bias);


    // layer3
    float inner3[fpn_output_channels * height_8 * width_8];
    Conv2d(layer3, inner3, "fpn.inner_blocks.2", channels_3, height_8, width_8, fpn_output_channels, height_8, width_8, inner_kernel_size, stride, inner_padding, groups, apply_bias);

    float top_down3[fpn_output_channels * height_8 * width_8];
    interpolate(inner4, top_down3, "nearest", fpn_output_channels, height_16, width_16, height_8, width_8);
    for (int idx = 0; idx < fpn_output_channels * height_8 * width_8; idx++)
        inner3[idx] += top_down3[idx];

    Conv2d(inner3, features_one_eight, "fpn.layer_blocks.2", fpn_output_channels, height_8, width_8, fpn_output_channels, height_8, width_8, layer_kernel_size, stride, layer_padding, groups, apply_bias);


    // layer2
    float inner2[fpn_output_channels * height_4 * width_4];
    Conv2d(layer2, inner2, "fpn.inner_blocks.1", channels_2, height_4, width_4, fpn_output_channels, height_4, width_4, inner_kernel_size, stride, inner_padding, groups, apply_bias);

    float top_down2[fpn_output_channels * height_4 * width_4];
    interpolate(inner3, top_down2, "nearest", fpn_output_channels, height_8, width_8, height_4, width_4);
    for (int idx = 0; idx < fpn_output_channels * height_4 * width_4; idx++)
        inner2[idx] += top_down2[idx];

    Conv2d(inner2, features_quarter, "fpn.layer_blocks.1", fpn_output_channels, height_4, width_4, fpn_output_channels, height_4, width_4, layer_kernel_size, stride, layer_padding, groups, apply_bias);


    // layer1
    float inner1[fpn_output_channels * height_2 * width_2];
    Conv2d(layer1, inner1, "fpn.inner_blocks.0", channels_1, height_2, width_2, fpn_output_channels, height_2, width_2, inner_kernel_size, stride, inner_padding, groups, apply_bias);

    float top_down1[fpn_output_channels * height_2 * width_2];
    interpolate(inner2, top_down1, "nearest", fpn_output_channels, height_4, width_4, height_2, width_2);
    for (int idx = 0; idx < fpn_output_channels * height_2 * width_2; idx++)
        inner1[idx] += top_down1[idx];

    Conv2d(inner1, features_half, "fpn.layer_blocks.0", fpn_output_channels, height_2, width_2, fpn_output_channels, height_2, width_2, layer_kernel_size, stride, layer_padding, groups, apply_bias);
}


void CostVolumeEncoder(const float features_half[fpn_output_channels * height_2 * width_2],
                       const float features_quarter[fpn_output_channels * height_4 * width_4],
                       const float features_one_eight[fpn_output_channels * height_8 * width_8],
                       const float features_one_sixteen[fpn_output_channels * height_16 * width_16],
                       const float cost_volume[n_depth_levels * height_2 * width_2],
                       float skip0[hyper_channels * height_2 * width_2],
                       float skip1[(hyper_channels * 2) * height_4 * width_4],
                       float skip2[(hyper_channels * 4) * height_8 * width_8],
                       float skip3[(hyper_channels * 8) * height_16 * width_16],
                       float bottom[(hyper_channels * 16) * height_32 * width_32]) {

    constexpr int stride = 1;
    constexpr bool apply_bn_relu = true;


    // 1st set
    constexpr int l0_in_channels = fpn_output_channels + n_depth_levels;
    constexpr int l0_in_height = height_2;
    constexpr int l0_in_width = width_2;

    constexpr int l0_kernel_size = 5;
    constexpr int l0_mid_channels = hyper_channels;
    float l0_in[l0_in_channels * l0_in_height * l0_in_width];
    for (int idx = 0; idx < fpn_output_channels * l0_in_height * l0_in_width; idx++)
        l0_in[idx] = features_half[idx];
    for (int idx = 0; idx < n_depth_levels * l0_in_height * l0_in_width; idx++)
        l0_in[idx + (fpn_output_channels * l0_in_height * l0_in_width)] = cost_volume[idx];
    conv_layer(l0_in, skip0, "aggregator0", l0_in_channels, l0_in_height, l0_in_width, l0_mid_channels, l0_in_height, l0_in_width, l0_kernel_size, stride, apply_bn_relu);

    constexpr int l0_out_channels = hyper_channels * 2;
    constexpr int l0_out_height = height_4;
    constexpr int l0_out_width = width_4;
    float l0_out[l0_out_channels * l0_out_height * l0_out_width];
    EncoderBlock(skip0, l0_out, "encoder_block0", l0_mid_channels, l0_in_height, l0_in_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size);


    // 2nd set
    constexpr int l1_in_channels = fpn_output_channels + l0_out_channels;
    constexpr int l1_in_height = height_4;
    constexpr int l1_in_width = width_4;

    constexpr int l1_kernel_size = 3;
    constexpr int l1_mid_channels = hyper_channels * 2;
    float l1_in[l1_in_channels * l1_in_height * l1_in_width];
    for (int idx = 0; idx < fpn_output_channels * l1_in_height * l1_in_width; idx++)
        l1_in[idx] = features_quarter[idx];
    for (int idx = 0; idx < l0_out_channels * l1_in_height * l1_in_width; idx++)
        l1_in[idx + (fpn_output_channels * l1_in_height * l1_in_width)] = l0_out[idx];
    conv_layer(l1_in, skip1, "aggregator1", l1_in_channels, l1_in_height, l1_in_width, l1_mid_channels, l1_in_height, l1_in_width, l1_kernel_size, stride, apply_bn_relu);

    constexpr int l1_out_channels = hyper_channels * 4;
    constexpr int l1_out_height = height_8;
    constexpr int l1_out_width = width_8;
    float l1_out[l1_out_channels * l1_out_height * l1_out_width];
    EncoderBlock(skip1, l1_out, "encoder_block1", l1_mid_channels, l1_in_height, l1_in_width, l1_out_channels, l1_out_height, l1_out_width, l1_kernel_size);


    // 3rd set
    constexpr int l2_in_channels = fpn_output_channels + l1_out_channels;
    constexpr int l2_in_height = height_8;
    constexpr int l2_in_width = width_8;

    constexpr int l2_kernel_size = 3;
    constexpr int l2_mid_channels = hyper_channels * 4;
    float l2_in[l2_in_channels * l2_in_height * l2_in_width];
    for (int idx = 0; idx < fpn_output_channels * l2_in_height * l2_in_width; idx++)
        l2_in[idx] = features_one_eight[idx];
    for (int idx = 0; idx < l1_out_channels * l2_in_height * l2_in_width; idx++)
        l2_in[idx + (fpn_output_channels * l2_in_height * l2_in_width)] = l1_out[idx];
    conv_layer(l2_in, skip2, "aggregator2", l2_in_channels, l2_in_height, l2_in_width, l2_mid_channels, l2_in_height, l2_in_width, l2_kernel_size, stride, apply_bn_relu);

    constexpr int l2_out_channels = hyper_channels * 8;
    constexpr int l2_out_height = height_16;
    constexpr int l2_out_width = width_16;
    float l2_out[l2_out_channels * l2_out_height * l2_out_width];
    EncoderBlock(skip2, l2_out, "encoder_block2", l2_mid_channels, l2_in_height, l2_in_width, l2_out_channels, l2_out_height, l2_out_width, l2_kernel_size);


    // 4th set
    constexpr int l3_in_channels = fpn_output_channels + l2_out_channels;
    constexpr int l3_in_height = height_16;
    constexpr int l3_in_width = width_16;

    constexpr int l3_kernel_size = 3;
    constexpr int l3_mid_channels = hyper_channels * 8;
    float l3_in[l3_in_channels * l3_in_height * l3_in_width];
    for (int idx = 0; idx < fpn_output_channels * l3_in_height * l3_in_width; idx++)
        l3_in[idx] = features_one_sixteen[idx];
    for (int idx = 0; idx < l2_out_channels * l3_in_height * l3_in_width; idx++)
        l3_in[idx + (fpn_output_channels * l3_in_height * l3_in_width)] = l2_out[idx];
    conv_layer(l3_in, skip3, "aggregator3", l3_in_channels, l3_in_height, l3_in_width, l3_mid_channels, l3_in_height, l3_in_width, l3_kernel_size, stride, apply_bn_relu);

    constexpr int l3_out_channels = hyper_channels * 16;
    constexpr int l3_out_height = height_32;
    constexpr int l3_out_width = width_32;
    EncoderBlock(skip3, bottom, "encoder_block3", l3_mid_channels, l3_in_height, l3_in_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size);
}


void CostVolumeDecoder(const float image[3 * test_image_height * test_image_width],
                       const float skip0[hyper_channels * height_2 * width_2],
                       const float skip1[(hyper_channels * 2) * height_4 * width_4],
                       const float skip2[(hyper_channels * 4) * height_8 * width_8],
                       const float skip3[(hyper_channels * 8) * height_16 * width_16],
                       const float bottom[(hyper_channels * 16) * height_32 * width_32],
                       float depth_full[test_image_height * test_image_width]) {

    constexpr bool apply_bn_relu = true;

    // 1st set
    constexpr int l0_in_channels = hyper_channels * 16;
    constexpr int l0_in_height = height_32;
    constexpr int l0_in_width = width_32;

    constexpr int l0_kernel_size = 3;
    constexpr bool l0_plus_one = false;
    constexpr int l0_out_channels = hyper_channels * 8;
    constexpr int l0_out_height = height_16;
    constexpr int l0_out_width = width_16;
    float* null_depth = nullptr;
    float decoder_block1[l0_out_channels * l0_out_height * l0_out_width];
    DecoderBlock(bottom, skip3, null_depth, decoder_block1, "decoder_block1", l0_in_channels, l0_in_height, l0_in_width, l0_kernel_size, apply_bn_relu, l0_plus_one);

    float sigmoid_depth_one_sixteen[1 * l0_out_height * l0_out_width];
    depth_layer_3x3(decoder_block1, sigmoid_depth_one_sixteen, "depth_layer_one_sixteen", l0_out_channels, l0_out_height, l0_out_width);


    // 2nd set
    constexpr int l1_kernel_size = 3;
    constexpr bool l1_plus_one = true;
    constexpr int l1_out_channels = hyper_channels * 4;
    constexpr int l1_out_height = height_8;
    constexpr int l1_out_width = width_8;
    float decoder_block2[l1_out_channels * l1_out_height * l1_out_width];
    DecoderBlock(decoder_block1, skip2, sigmoid_depth_one_sixteen, decoder_block2, "decoder_block2", l0_out_channels, l0_out_height, l0_out_width, l1_kernel_size, apply_bn_relu, l1_plus_one);

    float sigmoid_depth_one_eight[1 * l1_out_height * l1_out_width];
    depth_layer_3x3(decoder_block2, sigmoid_depth_one_eight, "depth_layer_one_eight", l1_out_channels, l1_out_height, l1_out_width);


    // 3rd set
    constexpr int l2_kernel_size = 3;
    constexpr bool l2_plus_one = true;
    constexpr int l2_out_channels = hyper_channels * 2;
    constexpr int l2_out_height = height_4;
    constexpr int l2_out_width = width_4;
    float decoder_block3[l2_out_channels * l2_out_height * l2_out_width];
    DecoderBlock(decoder_block2, skip1, sigmoid_depth_one_eight, decoder_block3, "decoder_block3", l1_out_channels, l1_out_height, l1_out_width, l2_kernel_size, apply_bn_relu, l2_plus_one);

    float sigmoid_depth_quarter[1 * l2_out_height * l2_out_width];
    depth_layer_3x3(decoder_block3, sigmoid_depth_quarter, "depth_layer_quarter", l2_out_channels, l2_out_height, l2_out_width);


    // 4th set
    constexpr int l3_kernel_size = 5;
    constexpr bool l3_plus_one = true;
    constexpr int l3_out_channels = hyper_channels;
    constexpr int l3_out_height = height_2;
    constexpr int l3_out_width = width_2;
    float decoder_block4[l3_out_channels * l3_out_height * l3_out_width];
    DecoderBlock(decoder_block3, skip0, sigmoid_depth_quarter, decoder_block4, "decoder_block4", l2_out_channels, l2_out_height, l2_out_width, l3_kernel_size, apply_bn_relu, l3_plus_one);

    float sigmoid_depth_half[1 * l3_out_height * l3_out_width];
    depth_layer_3x3(decoder_block4, sigmoid_depth_half, "depth_layer_half", l3_out_channels, l3_out_height, l3_out_width);


    // 5th set
    constexpr int l4_in_height = l3_out_height * 2;
    constexpr int l4_in_width = l3_out_width * 2;
    float scaled_depth[1 * l4_in_height * l4_in_width];
    interpolate(sigmoid_depth_half, scaled_depth, "bilinear", 1, l3_out_height, l3_out_width, l4_in_height, l4_in_width);

    float scaled_decoder[l3_out_channels * l4_in_height * l4_in_width];
    interpolate(decoder_block4, scaled_decoder, "bilinear", l3_out_channels, l3_out_height, l3_out_width, l4_in_height, l4_in_width);

    constexpr int l4_in_channels = l3_out_channels + 4;
    float scaled_combined[l4_in_channels * l4_in_height * l4_in_width];
    for (int idx = 0; idx < l3_out_channels * l4_in_height * l4_in_width; idx++)
        scaled_combined[idx] = scaled_decoder[idx];
    for (int idx = 0; idx < 1 * l4_in_height * l4_in_width; idx++)
        scaled_combined[idx + (l3_out_channels * l4_in_height * l4_in_width)] = scaled_depth[idx];
    for (int idx = 0; idx < 3 * l4_in_height * l4_in_width; idx++)
        scaled_combined[idx + ((l3_out_channels+1) * l4_in_height * l4_in_width)] = image[idx];

    constexpr int l4_kernel_size = 5;
    constexpr int l4_stride = 1;
    constexpr int l4_out_channels = hyper_channels;
    constexpr int l4_out_height = test_image_height;
    constexpr int l4_out_width = test_image_width;

    float refined0[l4_out_channels * l4_out_height * l4_out_width];
    conv_layer(scaled_combined, refined0, "refine.0", l4_in_channels, l4_in_height, l4_in_width, l4_out_channels, l4_out_height, l4_out_width, l4_kernel_size, l4_stride, apply_bn_relu);

    float refined1[l4_out_channels * l4_out_height * l4_out_width];
    conv_layer(refined0, refined1, "refine.1", l4_out_channels, l4_out_height, l4_out_width, l4_out_channels, l4_out_height, l4_out_width, l4_kernel_size, l4_stride, apply_bn_relu);

    depth_layer_3x3(refined1, depth_full, "depth_layer_full", l4_out_channels, l4_out_height, l4_out_width);

    for (int idx = 0; idx < l4_out_height * l4_out_width; idx++)
        depth_full[idx] = 1.0 / (inverse_depth_multiplier * depth_full[idx] + inverse_depth_base);

}


void LSTMFusion(const float current_encoding[(hyper_channels * 16) * height_32 * width_32],
                float hidden_state[hid_channels * height_32 * width_32],
                float cell_state[hid_channels * height_32 * width_32]) {

    constexpr int in_channels = hyper_channels * 16;
    constexpr int l0_in_channels = in_channels + hid_channels;
    constexpr int l0_out_channels = 4 * hid_channels;
    float combined[l0_in_channels * height_32 * width_32];
    for (int idx = 0; idx < in_channels * height_32 * width_32; idx++)
        combined[idx] = current_encoding[idx];
    for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
        combined[idx + (in_channels * height_32 * width_32)] = hidden_state[idx];

    constexpr int kernel_size = 3;
    constexpr int stride = 1;
    constexpr int padding = (kernel_size - 1) / 2;
    constexpr int groups = 1;
    constexpr bool apply_bias = false;
    float combined_conv[l0_out_channels * height_32 * width_32];
    Conv2d(combined, combined_conv, "lstm_cell.conv", l0_in_channels, height_32, width_32, l0_out_channels, height_32, width_32, kernel_size, stride, padding, groups, apply_bias);

    float* ii = combined_conv;
    float* ff = combined_conv + 1 * (hid_channels * height_32 * width_32);
    float* oo = combined_conv + 2 * (hid_channels * height_32 * width_32);
    float* gg = combined_conv + 3 * (hid_channels * height_32 * width_32);

    Sigmoid(ii, hid_channels, height_32, width_32);
    Sigmoid(ff, hid_channels, height_32, width_32);
    Sigmoid(oo, hid_channels, height_32, width_32);

    layer_norm(gg, hid_channels, height_32, width_32);
    celu(gg, hid_channels, height_32, width_32);

    for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
        cell_state[idx] = ff[idx] * cell_state[idx] + ii[idx] * gg[idx];

    layer_norm(cell_state, hid_channels, height_32, width_32);
    for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
        hidden_state[idx] = cell_state[idx];

    celu(hidden_state, hid_channels, height_32, width_32);
    for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
        hidden_state[idx] *= oo[idx];

}
