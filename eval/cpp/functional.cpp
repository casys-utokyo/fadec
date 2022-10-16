#include "config.h"

void layer_norm(float* x, const int channels, const int height, const int width) {

    constexpr float eps = 1e-5;
    const int n1 = height * width;
    for (int i = 0; i < channels; i++) {
        float e = 0.f;
        float v = 0.f;
        for (int idx = 0; idx < height * width; idx++) {
            e += x[i * (height * width) + idx];
            v += x[i * (height * width) + idx] * x[i * (height * width) + idx];
        }
        e /= n1;
        v /= n1;
        v -= e * e;
        for (int idx = 0; idx < height * width; idx++)
            x[i * (height * width) + idx] = (x[i * (height * width) + idx] - e) / sqrt(v + eps);
    }
}


void interpolate(const float* input, float* output, const string mode,
                const int channels, const int in_height, const int in_width,
                const int out_height, const int out_width) {

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
                const int y_int = floor(y);
                const int x_int = floor(x);
                const int ys[2] = {y_int, y_int + 1};
                const int xs[2] = {x_int, x_int + 1};
                const float dys[2] = {y - ys[0], ys[1] - y};
                const float dxs[2] = {x - xs[0], xs[1] - x};
                for (int i = 0; i < channels; i++) {
                    const int output_idx = (i * out_height + j) * out_width + k;
                    output[output_idx] = 0;
                    for (int yi = 0; yi < 2; yi++) for (int xi = 0; xi < 2; xi++) {
                        const int input_idx = (i * in_height + ys[yi]) * in_width + xs[xi];
                        output[output_idx] += dys[1-yi] * dxs[1-xi] * input[input_idx];
                    }
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


void grid_sample(const float* image, const float* warping, float* warped_image,
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
