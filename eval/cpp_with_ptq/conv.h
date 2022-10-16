#pragma once

void Conv2d(const qaint* input,
            qaint* output,
            const string param_path,
            const int in_channels, const int in_height, const int in_width,
            const int out_channels, const int out_height, const int out_width,
            const int kernel_size, const int stride, const int padding, const int groups,
            const bool apply_scale, const string activation,
            const int act_in, int& act_out) {

    // https://ichi.pro/conv-2-d-saigo-ni-fuxowa-do-pasu-de-nani-ga-okoru-ka-o-rikaisuru-30488625459528

    // to check order of layers
    // print1(param_path + ".weight");
    // if (apply_bias) print1(param_path + ".bias");

    const int wshift = w_shifts[conv_cnt];
    const qwint* weight = weights + w_idx[conv_cnt];
    const int bshift = b_shifts[conv_cnt];
    const qbint* bias = biases + b_idx[conv_cnt];

    const int xshift = cin_shifts[conv_cnt];
    const int yshift = cout_shifts[conv_cnt];
    conv_cnt++;

    const int sshift = apply_scale ? s_shifts[bn_cnt] : 0;
    const qsint* scale = apply_scale ? scales + s_idx[bn_cnt++] : nullptr;

    print_neg_shift(param_path, "wshift", wshift);
    print_neg_shift(param_path, "bshift", bshift);
    print_neg_shift(param_path, "bshift > 32", 32 - bshift);
    print_neg_shift(param_path, "xshift", xshift);
    print_neg_shift(param_path, "yshift", yshift);
    if (apply_scale) print_neg_shift(param_path, "sshift", sshift);
    print_neg_shift(param_path, "bshift - (xshift + wshift)", bshift - (xshift + wshift));
    print_neg_shift(param_path, "xshift + wshift + sshift - yshift", xshift + wshift + sshift - yshift);

    const int ocpg = out_channels / groups;
    const int icpg = in_channels / groups;
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < ocpg; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    const int och = g * ocpg + oc;
                    qmint sum = 0;

                    for (int ic = 0; ic < icpg; ic++) {
                        const int ich = g * icpg + ic;

                        for (int kh = 0; kh <= 2*padding; kh++) {
                            for (int kw = 0; kw <= 2*padding; kw++) {
                                const int ih = oh * stride + kh - padding;
                                const int iw = ow * stride + kw - padding;

                                const int input_idx = (ich * in_height + ih) * in_width + iw;
                                const int weight_idx = ((och * icpg + ic) * kernel_size + kh) * kernel_size + kw;

                                sum += (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) ? 0 :
                                       input[input_idx] * (qmint) weight[weight_idx];
                            }
                        }
                    }

                    sum += bias[och] >> (bshift - (xshift + wshift));
                    sum = apply_scale ? sum * scale[och] : sum;
                    const int output_idx = (och * out_height + oh) * out_width + ow;
                    output[output_idx] = clip(sum >> (xshift + wshift + sshift - yshift));
                }
            }
        }
    }

    if (activation == "relu") {
        ReLU(output, out_channels, out_height, out_width);
    } else if (activation == "sigmoid") {
        Sigmoid(output, out_channels, out_height, out_width);
    } else if (activation != "none") {
        print1(activation);
    }
}
