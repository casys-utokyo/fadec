#pragma once

void Conv2d(const float* input,
            float* output,
            const string param_path,
            const int in_channels, const int in_height, const int in_width,
            const int out_channels, const int out_height, const int out_width,
            const int kernel_size, const int stride, const int padding, const int groups, const bool apply_bias) {

    // https://ichi.pro/conv-2-d-saigo-ni-fuxowa-do-pasu-de-nani-ga-okoru-ka-o-rikaisuru-30488625459528
    const float* weight = params + start_idx[param_cnt++];
    const float* bias = apply_bias ? params + start_idx[param_cnt++] : nullptr;
    // print1(param_path + ".weight");
    // if (apply_bias) print1(param_path + ".bias");

    const int ocpg = out_channels / groups;
    const int icpg = in_channels / groups;
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < ocpg; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    const int och = g * ocpg + oc;
                    float sum = (apply_bias) ? bias[och] : 0.f;

                    for (int ic = 0; ic < icpg; ic++) {
                        const int ich = g * icpg + ic;

                        for (int kh = 0; kh <= 2*padding; kh++) {
                            for (int kw = 0; kw <= 2*padding; kw++) {
                                const int ih = oh * stride + kh - padding;
                                const int iw = ow * stride + kw - padding;

                                const int input_idx = (ich * in_height + ih) * in_width + iw;
                                const int weight_idx = ((och * icpg + ic) * kernel_size + kh) * kernel_size + kw;

                                sum += (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) ? 0.f :
                                        input[input_idx] * weight[weight_idx];
                            }
                        }
                    }

                    const int output_idx = (och * out_height + oh) * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}
