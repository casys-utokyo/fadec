#include "config.h"

void layer_norm(qaint* x, const int channels, const int height, const int width) {
    constexpr float eps = 1e-5;
    const int n1 = height * width;
    // const qaint* ln_ave = ln_aves + ln_idx[ln_cnt];
    // const qaint* ln_inv_std = ln_inv_stds + ln_idx[ln_cnt];
    for (int i = 0; i < channels; i++) {
        float e = 0;
        float v = 0;
        for (int idx = 0; idx < height * width; idx++) {
            e += x[i * (height * width) + idx];
            v += ((qmint) x[i * (height * width) + idx]) * x[i * (height * width) + idx];
        }
        e /= n1;
        v /= n1;
        v -= e * e;
        // if (i == 0) {
        //     print2(ln_ave[i], e);
        //     print2(((qmint) ln_inv_std[i]) / (float) (1 << lnin_shifts[ln_cnt]), 1.0 / sqrt(v + eps) * (1 << lnout_shifts[ln_cnt]));
        // }
        for (int idx = 0; idx < height * width; idx++)
            x[i * (height * width) + idx] = clip((x[i * (height * width) + idx] - e) / sqrt(v + eps) * (1 << lnout_shifts[ln_cnt]));
            // x[i * (height * width) + idx] = ((x[i * (height * width) + idx] - ln_ave[i]) * ((qmint) ln_inv_std[i])) >> lnin_shifts[ln_cnt];
    }
    ln_cnt++;
}


void add_layer(const qaint* x, qaint* y, const int layer_size, const string param_path, const int act_in0, const int act_in1, int& act_out) {
    const int xshift = ain1_shifts[add_cnt];
    const int yshift = ain2_shifts[add_cnt];
    const int outshift = aout_shifts[add_cnt];
    print_neg_shift(param_path, "xshift", xshift);
    print_neg_shift(param_path, "yshift", yshift);
    print_neg_shift(param_path, "outshift", outshift);
    add_cnt++;
    const int mshift = max(max(xshift, yshift), outshift);
    for (int idx = 0; idx < layer_size; idx++)
        y[idx] = clip((((qmint) y[idx] << (mshift - yshift)) + (((qmint) x[idx] << (mshift - xshift)))) >> (mshift - outshift));

}


void cat_layer(const qaint* x0, const qaint* x1, qaint* y,
               const int in_channels0, const int in_channels1, const int height, const int width,
               const int x0shift, const int x1shift,
               const string param_path, const int act_in0, const int act_in1, int& act_out) {

    for (int idx = 0; idx < in_channels0 * height * width; idx++)
        y[idx] = x0[idx] >> x0shift;
    for (int idx = 0; idx < in_channels1 * height * width; idx++)
        y[idx + (in_channels0 * height * width)] = x1[idx] >> x1shift;

}


void cat_layer(const qaint* x0, const qaint* x1, const qaint* x2, qaint* y,
               const int in_channels0, const int in_channels1, const int in_channels2, const int height, const int width,
               const int x0shift, const int x1shift, const int x2shift,
               const string param_path, const int act_in0, const int act_in1, const int act_in2, int& act_out) {

    for (int idx = 0; idx < in_channels0 * height * width; idx++)
        y[idx] = x0[idx] >> x0shift;
    for (int idx = 0; idx < in_channels1 * height * width; idx++)
        y[idx + (in_channels0 * height * width)] = x1[idx] >> x1shift;
    for (int idx = 0; idx < in_channels2 * height * width; idx++)
        y[idx + ((in_channels0 + in_channels1) * height * width)] = x2[idx] >> x2shift;

}


void interpolate(const qaint* input, qaint* output, const string mode,
                 const int channels, const int in_height, const int in_width,
                 const int out_height, const int out_width,
                 const int act_in, int& act_out) {

    // mode    fy  fx
    //  nearest 0.5 0.5
    //  nearest 0.5 0.5
    //  nearest 0.5 0.5
    //  nearest 0.5 0.5
    // bilinear 0.33333 0.4
    // bilinear 0.42857 0.45455
    // bilinear 0.42857 0.45455
    // bilinear 0.46667 0.47826
    // bilinear 0.46667 0.47826
    // bilinear 0.48387 0.48936
    // bilinear 0.48387 0.48936
    // bilinear 0.49206 0.49474
    // bilinear 0.49206 0.49474

    // const int xshift = oin_shifts[other_cnt];
    // const int yshift = oout_shifts[other_cnt];
    // other_cnt++;
    // if (yshift != xshift) print4("xshift and yshift differ in interpolation:", mode, xshift, yshift);

    if (mode == "nearest") {
        const float fy = (float) in_height / out_height;
        const float fx = (float) in_width / out_width;
        for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++) {
            const int y = j * fy;
            const int x = k * fx;
            for (int i = 0; i < channels; i++) {
                const int input_idx = (i * in_height + y) * in_width + x;
                const int output_idx = (i * out_height + j) * out_width + k;
                output[output_idx] = input[input_idx];
            }
        }
    } else if (mode == "bilinear") {
        if (in_height < out_height) {
            const float fy = (float) (in_height - 1) / (out_height - 1);
            const float fx = (float) (in_width - 1) / (out_width - 1);
            for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++) {
                const float y = j * fy;
                const float x = k * fx;
                const int y_int = (y < in_height-1) ? floor(y) : floor(y) - 1;
                const int x_int = (x < in_width-1) ? floor(x) : floor(x) - 1;
                const int ys[2] = {y_int, y_int + 1};
                const int xs[2] = {x_int, x_int + 1};
                const float dys[2] = {y - ys[0], ys[1] - y};
                const float dxs[2] = {x - xs[0], xs[1] - x};
                for (int i = 0; i < channels; i++) {
                    const int output_idx = (i * out_height + j) * out_width + k;
                    float sum = 0;
                    for (int yi = 0; yi < 2; yi++) for (int xi = 0; xi < 2; xi++) {
                        const int input_idx = (i * in_height + ys[yi]) * in_width + xs[xi];
                        sum += dys[1-yi] * dxs[1-xi] * input[input_idx];
                    }
                    // output[output_idx] = ((qaint) round(sum)) >> (xshift - yshift);
                    output[output_idx] = round(sum);
                }
            }
        } else {
            cout << "in_height is larger than out_height" << "\n";
            exit(1);
        }
    } else {
        cout << "The 'mode' option in interpolation should be 'nearest' or 'bilinear,' but it is " << mode << "\n";
        exit(1);
    }
}


void grid_sample(const qaint* image, const float* warping, float* warped_image,
                 const int channels, const int height, const int width) {

    for (int idx = 0; idx < height * width; idx++) {
        const float x = (warping[idx * 2 + 0] + 1) * (width - 1) / 2.0;
        const float y = (warping[idx * 2 + 1] + 1) * (height - 1) / 2.0;
        const int y_int = floor(y);
        const int x_int = floor(x);
        const int ys[2] = {y_int, y_int + 1};
        const int xs[2] = {x_int, x_int + 1};
        const float dys[2] = {y - ys[0], ys[1] - y};
        const float dxs[2] = {x - xs[0], xs[1] - x};
        for (int i = 0; i < channels; i++) {
            warped_image[i * (height * width) + idx] = 0;
            for (int yi = 0; yi < 2; yi++) for (int xi = 0; xi < 2; xi++) {
                const float val = (ys[yi] < 0 || height-1 < ys[yi] || xs[xi] < 0 || width-1 < xs[xi]) ? 0 : image[(i * height + ys[yi]) * width + xs[xi]];
                warped_image[i * (height * width) + idx] += dys[1-yi] * dxs[1-xi] * val;
            }
        }
    }
}
