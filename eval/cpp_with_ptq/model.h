#pragma once

void StandardLayer(const qaint* x, qaint* y, const string param_path,
                   const int channels, const int height, const int width,
                   const int kernel_size,
                   const int act_in, int& act_out) {

    constexpr int stride = 1;
    int act_out_conv;
    qaint y0[channels * height * width];
    conv_layer(x, y0, param_path + ".conv1", channels, height, width, channels, height, width, kernel_size, stride, act_in, act_out_conv);
    conv_layer(y0, y, param_path + ".conv2", channels, height, width, channels, height, width, kernel_size, stride, act_out_conv, act_out);
}


void DownconvolutionLayer(const qaint* x, qaint* y, const string param_path,
                          const int in_channels, const int in_height, const int in_width,
                          const int out_channels, const int out_height, const int out_width,
                          const int kernel_size,
                          const int act_in, int& act_out) {

    constexpr int stride = 2;
    conv_layer(x, y, param_path + ".down_conv", in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, stride, act_in, act_out);
}


void UpconvolutionLayer(const qaint* x, qaint* y, const string param_path,
                        const int in_channels, const int in_height, const int in_width,
                        const int out_channels, const int kernel_size,
                        const int act_in, int& act_out) {

    int act_out_up_x;
    const int out_height = in_height * 2;
    const int out_width = in_width * 2;
    qaint up_x[in_channels * out_height * out_width];
    interpolate(x, up_x, "bilinear", in_channels, in_height, in_width, out_height, out_width, act_in, act_out_up_x);

    constexpr int stride = 1;
    conv_layer(up_x, y, param_path + ".conv", in_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, act_out_up_x, act_out);
}


void EncoderBlock(const qaint* x, qaint* y, const string param_path,
                  const int in_channels, const int in_height, const int in_width,
                  const int out_channels, const int out_height, const int out_width,
                  const int kernel_size,
                  const int act_in, int& act_out) {

    qaint y0[out_channels * out_height * out_width];
    int act_out_dc;
    DownconvolutionLayer(x, y0, param_path + ".down_convolution", in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size, act_in, act_out_dc);
    StandardLayer(y0, y, param_path + ".standard_convolution", out_channels, out_height, out_width, kernel_size, act_out_dc, act_out);
}

int cat_cnt = 0;
void DecoderBlock(const qaint* x, const qaint* skip, const qaint* depth, qaint* y, const string param_path,
                  const int in_channels, const int in_height, const int in_width,
                  const int kernel_size, const bool plus_one,
                  const int act_in_x, const int act_in_skip, const int act_in_depth, int& act_out) {

    const int out_height = in_height * 2;
    const int out_width = in_width * 2;
    const int out_channels = in_channels / 2;

    int act_out_y0;
    qaint y0[out_channels * out_height * out_width];
    UpconvolutionLayer(x, y0, param_path + ".up_convolution", in_channels, in_height, in_width, out_channels, kernel_size, act_in_x, act_out_y0);

    // Aggregate skip and upsampled input
    const int l1_in_channels = plus_one ? in_channels + 1 : in_channels;
    qaint x1[l1_in_channels * out_height * out_width];
    // cat
    int act_out_x1;
    if (plus_one) {
        // 要注意: 1回目だけ 3 じゃなくて 2、interpolate はシフトしない
        int act_out_depth;
        interpolate(depth, x1 + (in_channels * out_height * out_width), "bilinear", 1, in_height, in_width, out_height, out_width, act_in_depth, act_out_depth);
        // for (int idx = 0; idx < out_channels * out_height * out_width; idx++)
        //     x1[idx] = y0[idx];
        // for (int idx = 0; idx < out_channels * out_height * out_width; idx++)
        //     x1[idx + (out_channels * out_height * out_width)] = skip[idx];
        // for (int idx = 0; idx < out_height * out_width; idx++)
        //     x1[idx + ((out_channels * 2) * out_height * out_width)] >>= 3;
        // const int rshift0 = (cat_cnt % 3 == 2) ? 1 : 0;
        // const int rshift1 = (cat_cnt % 3 == 1) ? 1 : 0;
        const int rshift = (cat_cnt % 3 == 0) ? 2 : 3;
        cat_layer(y0, skip, x1 + (in_channels * out_height * out_width), x1,
                  out_channels, out_channels, 1, out_height, out_width,
                //   rshift0, rshift1, 3, "cat6", act_out_y0, act_in_skip, act_out_depth, act_out_x1);
                  0, 0, rshift, "cat6", act_out_y0, act_in_skip, act_out_depth, act_out_x1);
        cat_cnt = ((cat_cnt) + 1) % 3;
    } else {
        // for (int idx = 0; idx < out_channels * out_height * out_width; idx++)
        //     x1[idx] = y0[idx] >> 1;
        // for (int idx = 0; idx < out_channels * out_height * out_width; idx++)
        //     x1[idx + (out_channels * out_height * out_width)] = skip[idx];
        cat_layer(y0, skip, x1, out_channels, out_channels, out_height, out_width,
                  1, 0, "cat5", act_out_y0, act_in_skip, act_out_x1);
    }
    // if (!plus_one) cin_shifts[conv_cnt]--; // 応急処置

    constexpr int stride = 1;

    int act_out_y1;
    qaint y1[out_channels * out_height * out_width];
    conv_layer(x1, y1, param_path + ".convolution1", l1_in_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, act_out_x1, act_out_y1);

    // Learn from aggregation
    conv_layer(y1, y, param_path + ".convolution2", out_channels, out_height, out_width, out_channels, out_height, out_width, kernel_size, stride, act_out_y1, act_out);
}


void FeatureExtractor(const qaint x[3 * test_image_height * test_image_width],
                      qaint layer1[channels_1 * height_2 * width_2],
                      qaint layer2[channels_2 * height_4 * width_4],
                      qaint layer3[channels_3 * height_8 * width_8],
                      qaint layer4[channels_4 * height_16 * width_16],
                      qaint layer5[channels_5 * height_32 * width_32],
                      const int act_in,
                      int& act_out_layer1,
                      int& act_out_layer2,
                      int& act_out_layer3,
                      int& act_out_layer4,
                      int& act_out_layer5) {

    constexpr int depths[8] = {32, channels_1, channels_2, channels_3, 80, channels_4, 192, channels_5};
    constexpr int apply_scale = true;

    // First layer: regular conv.
    constexpr int l0_kernel_size = 3;
    constexpr int l0_stride = 2;
    constexpr int l0_padding = 1;
    constexpr int l0_groups = 1;
    const string l0_activation = "relu";
    int l0_act_out;
    constexpr int l0_out_channels = depths[0];
    constexpr int l0_out_height = conv_out_size(test_image_height, l0_kernel_size, l0_stride, l0_padding);
    constexpr int l0_out_width = conv_out_size(test_image_width, l0_kernel_size, l0_stride, l0_padding);
    qaint y0[l0_out_channels * l0_out_height * l0_out_width];
    Conv2d(x, y0, "layer1.0", 3, test_image_height, test_image_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size, l0_stride, l0_padding, l0_groups, apply_scale, l0_activation, act_in, l0_act_out);

    // Depthwise separable, no skip.
    constexpr int l3_kernel_size = 3;
    constexpr int l3_stride = 1;
    constexpr int l3_padding = 1;
    constexpr int l3_groups = depths[0];
    const string l3_activation = "relu";
    int l3_act_out;
    constexpr int l3_out_channels = depths[0];
    constexpr int l3_out_height = conv_out_size(l0_out_height, l3_kernel_size, l3_stride, l3_padding);
    constexpr int l3_out_width = conv_out_size(l0_out_width, l3_kernel_size, l3_stride, l3_padding);
    qaint y3[l3_out_channels * l3_out_height * l3_out_width];
    Conv2d(y0, y3, "layer1.3", l0_out_channels, l0_out_height, l0_out_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size, l3_stride, l3_padding, l3_groups, apply_scale, l3_activation, l0_act_out, l3_act_out);

    constexpr int l6_kernel_size = 1;
    constexpr int l6_stride = 1;
    constexpr int l6_padding = 0;
    constexpr int l6_groups = 1;
    const string l6_activation = "none";
    constexpr int l6_out_channels = depths[1];
    constexpr int l6_out_height = conv_out_size(l3_out_height, l6_kernel_size, l6_stride, l6_padding);
    constexpr int l6_out_width = conv_out_size(l3_out_width, l6_kernel_size, l6_stride, l6_padding);
    Conv2d(y3, layer1, "layer1.6", l3_out_channels, l3_out_height, l3_out_width, l6_out_channels, l6_out_height, l6_out_width, l6_kernel_size, l6_stride, l6_padding, l6_groups, apply_scale, l6_activation, l3_act_out, act_out_layer1);

    // MNASNet blocks: stacks of inverted residuals.
    constexpr int l8_kernel_size = 3;
    constexpr int l8_stride = 2;
    constexpr int l8_expansion_factor = 3;
    constexpr int l8_repeats = 3;
    constexpr int l8_out_channels = depths[2];
    constexpr int l8_out_height = stack_out_size(l6_out_height, l8_kernel_size, l8_stride);
    constexpr int l8_out_width = stack_out_size(l6_out_width, l8_kernel_size, l8_stride);
    _stack(layer1, layer2, "layer2.0", l6_out_channels, l6_out_height, l6_out_width, l8_out_channels, l8_out_height, l8_out_width, l8_kernel_size, l8_stride, l8_expansion_factor, l8_repeats, act_out_layer1, act_out_layer2);

    constexpr int l9_kernel_size = 5;
    constexpr int l9_stride = 2;
    constexpr int l9_expansion_factor = 3;
    constexpr int l9_repeats = 3;
    constexpr int l9_out_channels = depths[3];
    constexpr int l9_out_height = stack_out_size(l8_out_height, l9_kernel_size, l9_stride);
    constexpr int l9_out_width = stack_out_size(l8_out_width, l9_kernel_size, l9_stride);
    _stack(layer2, layer3, "layer3.0", l8_out_channels, l8_out_height, l8_out_width, l9_out_channels, l9_out_height, l9_out_width, l9_kernel_size, l9_stride, l9_expansion_factor, l9_repeats, act_out_layer2, act_out_layer3);

    constexpr int l10_kernel_size = 5;
    constexpr int l10_stride = 2;
    constexpr int l10_expansion_factor = 6;
    constexpr int l10_repeats = 3;
    int l10_act_out;
    constexpr int l10_out_channels = depths[4];
    constexpr int l10_out_height = stack_out_size(l9_out_height, l10_kernel_size, l10_stride);
    constexpr int l10_out_width = stack_out_size(l9_out_width, l10_kernel_size, l10_stride);
    qaint y10[l10_out_channels * l10_out_height * l10_out_width];
    _stack(layer3, y10, "layer4.0", l9_out_channels, l9_out_height, l9_out_width, l10_out_channels, l10_out_height, l10_out_width, l10_kernel_size, l10_stride, l10_expansion_factor, l10_repeats, act_out_layer3, l10_act_out);

    constexpr int l11_kernel_size = 3;
    constexpr int l11_stride = 1;
    constexpr int l11_expansion_factor = 6;
    constexpr int l11_repeats = 2;
    constexpr int l11_out_channels = depths[5];
    constexpr int l11_out_height = stack_out_size(l10_out_height, l11_kernel_size, l11_stride);
    constexpr int l11_out_width = stack_out_size(l10_out_width, l11_kernel_size, l11_stride);
    _stack(y10, layer4, "layer4.1", l10_out_channels, l10_out_height, l10_out_width, l11_out_channels, l11_out_height, l11_out_width, l11_kernel_size, l11_stride, l11_expansion_factor, l11_repeats, l10_act_out, act_out_layer4);

    constexpr int l12_kernel_size = 5;
    constexpr int l12_stride = 2;
    constexpr int l12_expansion_factor = 6;
    constexpr int l12_repeats = 4;
    int l12_act_out;
    constexpr int l12_out_channels = depths[6];
    constexpr int l12_out_height = stack_out_size(l11_out_height, l12_kernel_size, l12_stride);
    constexpr int l12_out_width = stack_out_size(l11_out_width, l12_kernel_size, l12_stride);
    qaint y12[l12_out_channels * l12_out_height * l12_out_width];
    _stack(layer4, y12, "layer5.0", l11_out_channels, l11_out_height, l11_out_width, l12_out_channels, l12_out_height, l12_out_width, l12_kernel_size, l12_stride, l12_expansion_factor, l12_repeats, act_out_layer4, l12_act_out);

    constexpr int l13_kernel_size = 3;
    constexpr int l13_stride = 1;
    constexpr int l13_expansion_factor = 6;
    constexpr int l13_repeats = 1;
    constexpr int l13_out_channels = depths[7];
    constexpr int l13_out_height = stack_out_size(l12_out_height, l13_kernel_size, l13_stride);
    constexpr int l13_out_width = stack_out_size(l12_out_width, l13_kernel_size, l13_stride);
    _stack(y12, layer5, "layer5.1", l12_out_channels, l12_out_height, l12_out_width, l13_out_channels, l13_out_height, l13_out_width, l13_kernel_size, l13_stride, l13_expansion_factor, l13_repeats, l12_act_out, act_out_layer5);

}


void FeatureShrinker(const qaint layer1[channels_1 * height_2 * width_2],
                     const qaint layer2[channels_2 * height_4 * width_4],
                     const qaint layer3[channels_3 * height_8 * width_8],
                     const qaint layer4[channels_4 * height_16 * width_16],
                     const qaint layer5[channels_5 * height_32 * width_32],
                     qaint features_half[fpn_output_channels * height_2 * width_2],
                     qaint features_quarter[fpn_output_channels * height_4 * width_4],
                     qaint features_one_eight[fpn_output_channels * height_8 * width_8],
                     qaint features_one_sixteen[fpn_output_channels * height_16 * width_16],
                     const int act_out_layer1,
                     const int act_out_layer2,
                     const int act_out_layer3,
                     const int act_out_layer4,
                     const int act_out_layer5,
                     int& act_out_half,
                     int& act_out_quarter,
                     int& act_out_one_eight,
                     int& act_out_one_sixteen) {

    // Module that adds a FPN from on top of a set of feature maps. This is based on
    // `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    // The feature maps are currently supposed to be in increasing depth order.
    // The input to the model is expected to be an OrderedDict[Tensor], containing
    // the feature maps on top of which the FPN will be added.

    constexpr int stride = 1;
    constexpr int groups = 1;
    constexpr bool apply_scale = false;
    const string activation = "none";

    constexpr int inner_kernel_size = 1;
    constexpr int inner_padding = 0;
    constexpr int layer_kernel_size = 3;
    constexpr int layer_padding = 1;

    // layer5
    qaint inner5[fpn_output_channels * height_32 * width_32];
    int act_in_inner5;
    Conv2d(layer5, inner5, "fpn.inner_blocks.4", channels_5, height_32, width_32, fpn_output_channels, height_32, width_32, inner_kernel_size, stride, inner_padding, groups, apply_scale, activation, act_out_layer5, act_in_inner5);

    // layer4 (order of interpolation and first conv is reversed from original)
    qaint top_down4[fpn_output_channels * height_16 * width_16];
    int act_out_top_down4;
    interpolate(inner5, top_down4, "nearest", fpn_output_channels, height_32, width_32, height_16, width_16, act_in_inner5, act_out_top_down4);

    qaint inner4[fpn_output_channels * height_16 * width_16];
    int act_in_inner4;
    Conv2d(layer4, inner4, "fpn.inner_blocks.3", channels_4, height_16, width_16, fpn_output_channels, height_16, width_16, inner_kernel_size, stride, inner_padding, groups, apply_scale, activation, act_out_layer4, act_in_inner4);

    const int layer_size4 = fpn_output_channels * height_16 * width_16;
    int act_out_inner4;
    add_layer(top_down4, inner4, layer_size4, "top_inner4", act_out_top_down4, act_in_inner4, act_out_inner4);
    Conv2d(inner4, features_one_sixteen, "fpn.layer_blocks.3", fpn_output_channels, height_16, width_16, fpn_output_channels, height_16, width_16, layer_kernel_size, stride, layer_padding, groups, apply_scale, activation, act_out_inner4, act_out_one_sixteen);

    // layer3
    qaint top_down3[fpn_output_channels * height_8 * width_8];
    int act_out_top_down3;
    interpolate(inner4, top_down3, "nearest", fpn_output_channels, height_16, width_16, height_8, width_8, act_out_inner4, act_out_top_down3);
    qaint inner3[fpn_output_channels * height_8 * width_8];
    int act_in_inner3;
    Conv2d(layer3, inner3, "fpn.inner_blocks.2", channels_3, height_8, width_8, fpn_output_channels, height_8, width_8, inner_kernel_size, stride, inner_padding, groups, apply_scale, activation, act_out_layer3, act_in_inner3);

    const int layer_size3 = fpn_output_channels * height_8 * width_8;
    int act_out_inner3;
    add_layer(top_down3, inner3, layer_size3, "top_inner3", act_out_top_down3, act_in_inner3, act_out_inner3);
    Conv2d(inner3, features_one_eight, "fpn.layer_blocks.2", fpn_output_channels, height_8, width_8, fpn_output_channels, height_8, width_8, layer_kernel_size, stride, layer_padding, groups, apply_scale, activation, act_out_inner3, act_out_one_eight);


    // layer2
    qaint top_down2[fpn_output_channels * height_4 * width_4];
    int act_out_top_down2;
    interpolate(inner3, top_down2, "nearest", fpn_output_channels, height_8, width_8, height_4, width_4, act_out_inner3, act_out_top_down2);
    qaint inner2[fpn_output_channels * height_4 * width_4];
    int act_in_inner2;
    Conv2d(layer2, inner2, "fpn.inner_blocks.1", channels_2, height_4, width_4, fpn_output_channels, height_4, width_4, inner_kernel_size, stride, inner_padding, groups, apply_scale, activation, act_out_layer2, act_in_inner2);

    const int layer_size2 = fpn_output_channels * height_4 * width_4;
    int act_out_inner2;
    add_layer(top_down2, inner2, layer_size2, "top_inner2", act_out_top_down2, act_in_inner2, act_out_inner2);
    Conv2d(inner2, features_quarter, "fpn.layer_blocks.1", fpn_output_channels, height_4, width_4, fpn_output_channels, height_4, width_4, layer_kernel_size, stride, layer_padding, groups, apply_scale, activation, act_out_inner2, act_out_quarter);


    // layer1
    qaint top_down1[fpn_output_channels * height_2 * width_2];
    int act_out_top_down1;
    interpolate(inner2, top_down1, "nearest", fpn_output_channels, height_4, width_4, height_2, width_2, act_out_inner2, act_out_top_down1);
    qaint inner1[fpn_output_channels * height_2 * width_2];
    int act_in_inner1;
    Conv2d(layer1, inner1, "fpn.inner_blocks.0", channels_1, height_2, width_2, fpn_output_channels, height_2, width_2, inner_kernel_size, stride, inner_padding, groups, apply_scale, activation, act_out_layer1, act_in_inner1);

    const int layer_size1 = fpn_output_channels * height_2 * width_2;
    int act_out_inner1;
    add_layer(top_down1, inner1, layer_size1, "top_inner1", act_out_top_down1, act_in_inner1, act_out_inner1);
    Conv2d(inner1, features_half, "fpn.layer_blocks.0", fpn_output_channels, height_2, width_2, fpn_output_channels, height_2, width_2, layer_kernel_size, stride, layer_padding, groups, apply_scale, activation, act_out_inner1, act_out_half);

}


void CostVolumeEncoder(const qaint features_half[fpn_output_channels * height_2 * width_2],
                       const qaint features_quarter[fpn_output_channels * height_4 * width_4],
                       const qaint features_one_eight[fpn_output_channels * height_8 * width_8],
                       const qaint features_one_sixteen[fpn_output_channels * height_16 * width_16],
                       const qaint cost_volume[n_depth_levels * height_2 * width_2],
                       qaint skip0[hyper_channels * height_2 * width_2],
                       qaint skip1[(hyper_channels * 2) * height_4 * width_4],
                       qaint skip2[(hyper_channels * 4) * height_8 * width_8],
                       qaint skip3[(hyper_channels * 8) * height_16 * width_16],
                       qaint bottom[(hyper_channels * 16) * height_32 * width_32],
                       const string filename,
                       const int act_out_half,
                       const int act_out_quarter,
                       const int act_out_one_eight,
                       const int act_out_one_sixteen,
                       const int act_out_cost_volume,
                       int& act_out_skip0,
                       int& act_out_skip1,
                       int& act_out_skip2,
                       int& act_out_skip3,
                       int& act_out_bottom) {

    constexpr int stride = 1;

    // 1st set
    constexpr int l0_in_channels = fpn_output_channels + n_depth_levels;
    constexpr int l0_in_height = height_2;
    constexpr int l0_in_width = width_2;

    constexpr int l0_kernel_size = 5;
    constexpr int l0_mid_channels = hyper_channels;
    qaint l0_in[l0_in_channels * l0_in_height * l0_in_width];
    // cat
    // for (int idx = 0; idx < fpn_output_channels * l0_in_height * l0_in_width; idx++)
    //     l0_in[idx] = features_half[idx] >> 2;
    // for (int idx = 0; idx < n_depth_levels * l0_in_height * l0_in_width; idx++)
    //     l0_in[idx + (fpn_output_channels * l0_in_height * l0_in_width)] = cost_volume[idx];
    int act_out_l0_in;
    cat_layer(features_half, cost_volume, l0_in, fpn_output_channels, n_depth_levels, l0_in_height, l0_in_width,
              2, 0, "cat0", act_out_half, act_out_cost_volume, act_out_l0_in);
    conv_layer(l0_in, skip0, "aggregator0", l0_in_channels, l0_in_height, l0_in_width, l0_mid_channels, l0_in_height, l0_in_width, l0_kernel_size, stride, act_out_l0_in, act_out_skip0);

    int act_out_l0_out;
    constexpr int l0_out_channels = hyper_channels * 2;
    constexpr int l0_out_height = height_4;
    constexpr int l0_out_width = width_4;
    qaint l0_out[l0_out_channels * l0_out_height * l0_out_width];
    EncoderBlock(skip0, l0_out, "encoder_block0", l0_mid_channels, l0_in_height, l0_in_width, l0_out_channels, l0_out_height, l0_out_width, l0_kernel_size, act_out_skip0, act_out_l0_out);


    // 2nd set
    constexpr int l1_in_channels = fpn_output_channels + l0_out_channels;
    constexpr int l1_in_height = height_4;
    constexpr int l1_in_width = width_4;

    constexpr int l1_kernel_size = 3;
    constexpr int l1_mid_channels = hyper_channels * 2;
    qaint l1_in[l1_in_channels * l1_in_height * l1_in_width];
    // cat
    // for (int idx = 0; idx < fpn_output_channels * l1_in_height * l1_in_width; idx++)
    //     l1_in[idx] = features_quarter[idx];
    // for (int idx = 0; idx < l0_out_channels * l1_in_height * l1_in_width; idx++)
    //     l1_in[idx + (fpn_output_channels * l1_in_height * l1_in_width)] = l0_out[idx] >> 3;
    // // cin_shifts[conv_cnt]--; // 応急処置
    int act_out_l1_in;
    cat_layer(features_quarter, l0_out, l1_in, fpn_output_channels, l0_out_channels, l1_in_height, l1_in_width,
              0, 3, "cat1", act_out_quarter, act_out_l0_out, act_out_l1_in);
            //   0, 2, "cat1", act_out_quarter, act_out_l0_out, act_out_l1_in);
    conv_layer(l1_in, skip1, "aggregator1", l1_in_channels, l1_in_height, l1_in_width, l1_mid_channels, l1_in_height, l1_in_width, l1_kernel_size, stride, act_out_l1_in, act_out_skip1);

    int act_out_l1_out;
    constexpr int l1_out_channels = hyper_channels * 4;
    constexpr int l1_out_height = height_8;
    constexpr int l1_out_width = width_8;
    qaint l1_out[l1_out_channels * l1_out_height * l1_out_width];
    EncoderBlock(skip1, l1_out, "encoder_block1", l1_mid_channels, l1_in_height, l1_in_width, l1_out_channels, l1_out_height, l1_out_width, l1_kernel_size, act_out_skip1, act_out_l1_out);


    // 3rd set
    constexpr int l2_in_channels = fpn_output_channels + l1_out_channels;
    constexpr int l2_in_height = height_8;
    constexpr int l2_in_width = width_8;

    constexpr int l2_kernel_size = 3;
    constexpr int l2_mid_channels = hyper_channels * 4;
    qaint l2_in[l2_in_channels * l2_in_height * l2_in_width];
    // cat
    // for (int idx = 0; idx < fpn_output_channels * l2_in_height * l2_in_width; idx++)
    //     l2_in[idx] = features_one_eight[idx];
    // for (int idx = 0; idx < l1_out_channels * l2_in_height * l2_in_width; idx++)
    //     l2_in[idx + (fpn_output_channels * l2_in_height * l2_in_width)] = l1_out[idx] >> 1;
    int act_out_l2_in;
    cat_layer(features_one_eight, l1_out, l2_in, fpn_output_channels, l1_out_channels, l2_in_height, l2_in_width,
              0, 1, "cat2", act_out_one_eight, act_out_l1_out, act_out_l2_in);
            //   0, 2, "cat2", act_out_one_eight, act_out_l1_out, act_out_l2_in);
    conv_layer(l2_in, skip2, "aggregator2", l2_in_channels, l2_in_height, l2_in_width, l2_mid_channels, l2_in_height, l2_in_width, l2_kernel_size, stride, act_out_l2_in, act_out_skip2);

    int act_out_l2_out;
    constexpr int l2_out_channels = hyper_channels * 8;
    constexpr int l2_out_height = height_16;
    constexpr int l2_out_width = width_16;
    qaint l2_out[l2_out_channels * l2_out_height * l2_out_width];
    EncoderBlock(skip2, l2_out, "encoder_block2", l2_mid_channels, l2_in_height, l2_in_width, l2_out_channels, l2_out_height, l2_out_width, l2_kernel_size, act_out_skip2, act_out_l2_out);


    // 4th set
    constexpr int l3_in_channels = fpn_output_channels + l2_out_channels;
    constexpr int l3_in_height = height_16;
    constexpr int l3_in_width = width_16;

    constexpr int l3_kernel_size = 3;
    constexpr int l3_mid_channels = hyper_channels * 8;
    qaint l3_in[l3_in_channels * l3_in_height * l3_in_width];
    // cat
    // for (int idx = 0; idx < fpn_output_channels * l3_in_height * l3_in_width; idx++)
    //     l3_in[idx] = features_one_sixteen[idx];
    // for (int idx = 0; idx < l2_out_channels * l3_in_height * l3_in_width; idx++)
    //     l3_in[idx + (fpn_output_channels * l3_in_height * l3_in_width)] = l2_out[idx] >> 1;
    // // cin_shifts[conv_cnt]--; // 応急処置
    int act_out_l3_in;
    cat_layer(features_one_sixteen, l2_out, l3_in, fpn_output_channels, l2_out_channels, l3_in_height, l3_in_width,
              0, 1, "cat3", act_out_one_sixteen, act_out_l2_out, act_out_l3_in);
    conv_layer(l3_in, skip3, "aggregator3", l3_in_channels, l3_in_height, l3_in_width, l3_mid_channels, l3_in_height, l3_in_width, l3_kernel_size, stride, act_out_l3_in, act_out_skip3);

    constexpr int l3_out_channels = hyper_channels * 16;
    constexpr int l3_out_height = height_32;
    constexpr int l3_out_width = width_32;
    EncoderBlock(skip3, bottom, "encoder_block3", l3_mid_channels, l3_in_height, l3_in_width, l3_out_channels, l3_out_height, l3_out_width, l3_kernel_size, act_out_skip3, act_out_bottom);

}


void CostVolumeDecoder(const qaint image[3 * test_image_height * test_image_width],
                       const qaint skip0[hyper_channels * height_2 * width_2],
                       const qaint skip1[(hyper_channels * 2) * height_4 * width_4],
                       const qaint skip2[(hyper_channels * 4) * height_8 * width_8],
                       const qaint skip3[(hyper_channels * 8) * height_16 * width_16],
                       const qaint bottom[(hyper_channels * 16) * height_32 * width_32],
                       qaint depth_full[test_image_height * test_image_width],
                       const int act_in,
                       const int act_out_skip0,
                       const int act_out_skip1,
                       const int act_out_skip2,
                       const int act_out_skip3,
                       const int act_out_bottom,
                       int& act_out_depth_full) {

    // 1st set
    constexpr int l0_in_channels = hyper_channels * 16;
    constexpr int l0_in_height = height_32;
    constexpr int l0_in_width = width_32;

    constexpr int l0_kernel_size = 3;
    constexpr bool l0_plus_one = false;
    int act_out_decoder_block1;
    constexpr int l0_out_channels = hyper_channels * 8;
    constexpr int l0_out_height = height_16;
    constexpr int l0_out_width = width_16;
    qaint* null_depth = nullptr;
    qaint decoder_block1[l0_out_channels * l0_out_height * l0_out_width];
    DecoderBlock(bottom, skip3, null_depth, decoder_block1, "decoder_block1", l0_in_channels, l0_in_height, l0_in_width, l0_kernel_size, l0_plus_one, act_out_bottom, act_out_skip3, 0, act_out_decoder_block1);

    int act_out_depth_one_sixteen;
    qaint sigmoid_depth_one_sixteen[1 * l0_out_height * l0_out_width];
    depth_layer_3x3(decoder_block1, sigmoid_depth_one_sixteen, "depth_layer_one_sixteen", l0_out_channels, l0_out_height, l0_out_width, act_out_decoder_block1, act_out_depth_one_sixteen);


    // 2nd set
    constexpr int l1_kernel_size = 3;
    constexpr bool l1_plus_one = true;
    int act_out_decoder_block2;
    constexpr int l1_out_channels = hyper_channels * 4;
    constexpr int l1_out_height = height_8;
    constexpr int l1_out_width = width_8;
    qaint decoder_block2[l1_out_channels * l1_out_height * l1_out_width];
    DecoderBlock(decoder_block1, skip2, sigmoid_depth_one_sixteen, decoder_block2, "decoder_block2", l0_out_channels, l0_out_height, l0_out_width, l1_kernel_size, l1_plus_one, act_out_decoder_block1, act_out_skip2, act_out_depth_one_sixteen, act_out_decoder_block2);

    int act_out_depth_one_eight;
    qaint sigmoid_depth_one_eight[1 * l1_out_height * l1_out_width];
    depth_layer_3x3(decoder_block2, sigmoid_depth_one_eight, "depth_layer_one_eight", l1_out_channels, l1_out_height, l1_out_width, act_out_decoder_block2, act_out_depth_one_eight);


    // 3rd set
    constexpr int l2_kernel_size = 3;
    constexpr bool l2_plus_one = true;
    int act_out_decoder_block3;
    constexpr int l2_out_channels = hyper_channels * 2;
    constexpr int l2_out_height = height_4;
    constexpr int l2_out_width = width_4;
    qaint decoder_block3[l2_out_channels * l2_out_height * l2_out_width];
    DecoderBlock(decoder_block2, skip1, sigmoid_depth_one_eight, decoder_block3, "decoder_block3", l1_out_channels, l1_out_height, l1_out_width, l2_kernel_size, l2_plus_one, act_out_decoder_block2, act_out_skip1, act_out_depth_one_eight, act_out_decoder_block3);

    int act_out_depth_quarter;
    qaint sigmoid_depth_quarter[1 * l2_out_height * l2_out_width];
    depth_layer_3x3(decoder_block3, sigmoid_depth_quarter, "depth_layer_quarter", l2_out_channels, l2_out_height, l2_out_width, act_out_decoder_block3, act_out_depth_quarter);


    // 4th set
    constexpr int l3_kernel_size = 5;
    constexpr bool l3_plus_one = true;
    int act_out_decoder_block4;
    constexpr int l3_out_channels = hyper_channels;
    constexpr int l3_out_height = height_2;
    constexpr int l3_out_width = width_2;
    qaint decoder_block4[l3_out_channels * l3_out_height * l3_out_width];
    DecoderBlock(decoder_block3, skip0, sigmoid_depth_quarter, decoder_block4, "decoder_block4", l2_out_channels, l2_out_height, l2_out_width, l3_kernel_size, l3_plus_one, act_out_decoder_block3, act_out_skip0, act_out_depth_quarter, act_out_decoder_block4);

    int act_out_depth_half;
    qaint sigmoid_depth_half[1 * l3_out_height * l3_out_width];
    depth_layer_3x3(decoder_block4, sigmoid_depth_half, "depth_layer_half", l3_out_channels, l3_out_height, l3_out_width, act_out_decoder_block4, act_out_depth_half);


    // 5th set
    int act_out_scaled_depth;
    constexpr int l4_in_height = l3_out_height * 2;
    constexpr int l4_in_width = l3_out_width * 2;
    qaint scaled_depth[1 * l4_in_height * l4_in_width];
    interpolate(sigmoid_depth_half, scaled_depth, "bilinear", 1, l3_out_height, l3_out_width, l4_in_height, l4_in_width, act_out_depth_half, act_out_scaled_depth);

    int act_out_scaled_decoder;
    qaint scaled_decoder[l3_out_channels * l4_in_height * l4_in_width];
    interpolate(decoder_block4, scaled_decoder, "bilinear", l3_out_channels, l3_out_height, l3_out_width, l4_in_height, l4_in_width, act_out_decoder_block4, act_out_scaled_decoder);

    constexpr int l4_in_channels = l3_out_channels + 4;
    qaint scaled_combined[l4_in_channels * l4_in_height * l4_in_width];
    // cat
    // for (int idx = 0; idx < l3_out_channels * l4_in_height * l4_in_width; idx++)
    //     scaled_combined[idx] = scaled_decoder[idx] >> 2;
    // for (int idx = 0; idx < 1 * l4_in_height * l4_in_width; idx++)
    //     scaled_combined[idx + (l3_out_channels * l4_in_height * l4_in_width)] = scaled_depth[idx] >> 4;
    // for (int idx = 0; idx < 3 * l4_in_height * l4_in_width; idx++)
    //     scaled_combined[idx + ((l3_out_channels+1) * l4_in_height * l4_in_width)] = image[idx];
    // cin_shifts[conv_cnt]--; // 応急処置
    int act_out_scaled_combined;
    cat_layer(scaled_decoder, scaled_depth, image, scaled_combined,
              l3_out_channels, 1, 3, l4_in_height, l4_in_width,
              2, 4, 0, "cat7", act_out_scaled_decoder, act_out_scaled_depth, act_in, act_out_scaled_combined);
            //   1, 3, 0, "cat7", act_out_scaled_decoder, act_out_scaled_depth, act_in, act_out_scaled_combined);

    constexpr int l4_kernel_size = 5;
    constexpr int l4_stride = 1;
    int act_out_refined0;
    constexpr int l4_out_channels = hyper_channels;
    constexpr int l4_out_height = test_image_height;
    constexpr int l4_out_width = test_image_width;
    qaint refined0[l4_out_channels * l4_out_height * l4_out_width];
    conv_layer(scaled_combined, refined0, "refine.0", l4_in_channels, l4_in_height, l4_in_width, l4_out_channels, l4_out_height, l4_out_width, l4_kernel_size, l4_stride, act_out_scaled_combined, act_out_refined0);

    int act_out_refined1;
    qaint refined1[l4_out_channels * l4_out_height * l4_out_width];
    conv_layer(refined0, refined1, "refine.1", l4_out_channels, l4_out_height, l4_out_width, l4_out_channels, l4_out_height, l4_out_width, l4_kernel_size, l4_stride, act_out_refined0, act_out_refined1);

    depth_layer_3x3(refined1, depth_full, "depth_layer_full", l4_out_channels, l4_out_height, l4_out_width, act_out_refined1, act_out_depth_full);
}


void LSTMFusion(const qaint current_encoding[(hyper_channels * 16) * height_32 * width_32],
                qaint hidden_state[hid_channels * height_32 * width_32],
                qaint cell_state[hid_channels * height_32 * width_32],
                const string filename,
                const int act_out_current_encoding,
                int& act_out_hidden_state,
                int& act_out_cell_state) {

    const int act_in_hidden_state = act_cnt++;

    constexpr int in_channels = hyper_channels * 16;
    constexpr int l0_in_channels = in_channels + hid_channels;
    constexpr int l0_out_channels = 4 * hid_channels;
    qaint combined[l0_in_channels * height_32 * width_32];
    // cat
    // for (int idx = 0; idx < in_channels * height_32 * width_32; idx++)
    //     combined[idx] = current_encoding[idx];
    // for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
    //     combined[idx + (in_channels * height_32 * width_32)] = hidden_state[idx] >> 1;
    // 要注意
    int act_out_combined;
    cat_layer(current_encoding, hidden_state, combined, in_channels, hid_channels, height_32, width_32,
              0, 0, "cat4", act_out_current_encoding, act_in_hidden_state, act_out_combined);
            //   0, 1, "cat4", act_out_current_encoding, act_in_hidden_state, act_out_combined);

    constexpr int kernel_size = 3;
    constexpr int stride = 1;
    constexpr int padding = (kernel_size - 1) / 2;
    constexpr int groups = 1;
    constexpr bool apply_scale = false;
    const string activation = "none";
    int act_out_combined_conv;
    qaint combined_conv[l0_out_channels * height_32 * width_32];
    Conv2d(combined, combined_conv, "lstm_cell.conv", l0_in_channels, height_32, width_32, l0_out_channels, height_32, width_32, kernel_size, stride, padding, groups, apply_scale, activation, act_out_combined, act_out_combined_conv);

    qaint* ii = combined_conv;
    qaint* ff = combined_conv + 1 * (hid_channels * height_32 * width_32);
    qaint* oo = combined_conv + 2 * (hid_channels * height_32 * width_32);
    qaint* gg = combined_conv + 3 * (hid_channels * height_32 * width_32);

    Sigmoid(ii, hid_channels, height_32, width_32);
    Sigmoid(ff, hid_channels, height_32, width_32);
    Sigmoid(oo, hid_channels, height_32, width_32);

    layer_norm(gg, hid_channels, height_32, width_32);
    celu(gg, hid_channels, height_32, width_32);

    // 要注意
    constexpr int mulshift = 2;
    constexpr int sumshift = (sigshift - mulshift) + celushift - cellshift;
    for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
        cell_state[idx] = clip(((((qmint) ff[idx] >> mulshift) * cell_state[idx]) + ((qmint) ii[idx] >> mulshift) * gg[idx]) >> sumshift);

    layer_norm(cell_state, hid_channels, height_32, width_32);
    for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
        hidden_state[idx] = cell_state[idx];

    celu(hidden_state, hid_channels, height_32, width_32);
    for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
        hidden_state[idx] = clip((hidden_state[idx] * (qmint) oo[idx]) >> (celushift + sigshift - oin_shifts[other_cnt]));

}
