#pragma once

void layer_norm(qaint* x, const int channels, const int height, const int width);

void add_layer(const qaint* x, qaint* y, const int layer_size, const string param_path, const int act_in0, const int act_in1, int& act_out);

void cat_layer(const qaint* x0, const qaint* x1, qaint* y,
               const int in_channels0, const int in_channels1, const int height, const int width,
               const int x0shift, const int x1shift,
               const string param_path, const int act_in0, const int act_in1, int& act_out);

void cat_layer(const qaint* x0, const qaint* x1, const qaint* x2, qaint* y,
               const int in_channels0, const int in_channels1, const int in_channels2, const int height, const int width,
               const int x0shift, const int x1shift, const int x2shift,
               const string param_path, const int act_in0, const int act_in1, const int act_in2, int& act_out);

template<int channels, int in_height, int in_width, int out_height, int out_width>
void interpolate(float*** input, float output[channels][out_height][out_width]) {
    if (in_height > out_height) {
        const float fy = (float) in_height / out_height;
        const float fx = (float) in_width / out_width;
        for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++) {
            const float y = j * fy + (fy - 1) / 2;
            const float x = k * fx + (fx - 1) / 2;
            const int y_int = floor(y);
            const int x_int = floor(x);
            const int ys[2] = {y_int, y_int + 1};
            const int xs[2] = {x_int, x_int + 1};
            const float dys[2] = {y - ys[0], ys[1] - y};
            const float dxs[2] = {x - xs[0], xs[1] - x};
            for (int i = 0; i < channels; i++) {
                output[i][j][k] = 0;
                for (int yi = 0; yi < 2; yi++) for (int xi = 0; xi < 2; xi++) {
                    const float val = input[i][min(in_height-1, max(0, ys[yi]))][min(in_width-1, max(0, xs[xi]))];
                    output[i][j][k] += dys[1-yi] * dxs[1-xi] * val;
                }
            }
        }
    } else {
        cout << "in_height is smaller than out_height" << "\n";
        exit(1);
    }
}


template<int channels, int in_height, int in_width, int out_height, int out_width>
void interpolate(const float input[channels][in_height][in_width], float output[channels][out_height][out_width], const string mode = "nearest") {
    if (mode == "nearest") {
        const float fy = (float) in_height / out_height;
        const float fx = (float) in_width / out_width;
        for (int j = 0; j < out_height; j++) for (int k = 0; k < out_width; k++) {
            const int y = j * fy;
            const int x = k * fx;
            for (int i = 0; i < channels; i++)
                output[i][j][k] = input[i][y][x];
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
                    output[i][j][k] = 0;
                    for (int yi = 0; yi < 2; yi++) for (int xi = 0; xi < 2; xi++)
                        output[i][j][k] += dys[1-yi] * dxs[1-xi] * input[i][ys[yi]][xs[xi]];
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

void interpolate(const qaint* input, qaint* output, const string mode,
                 const int channels, const int in_height, const int in_width,
                 const int out_height, const int out_width,
                 const int act_in, int& act_out);

void grid_sample(const qaint* image, const float* warping, float* warped_image,
                 const int channels, const int height, const int width);

template <int channels, int height, int width>
void grid_sample(const float image[channels][height][width],
                 const float warping[height][width][2],
                 float warped_image[channels][height][width]) {

    for (int j = 0; j < height; j++) for (int k = 0; k < width; k++) {
        const float x = (warping[j][k][0] + 1) * (width - 1) / 2.0;
        const float y = (warping[j][k][1] + 1) * (height - 1) / 2.0;
        const int y_int = floor(y);
        const int x_int = floor(x);
        const int ys[2] = {y_int, y_int + 1};
        const int xs[2] = {x_int, x_int + 1};
        const float dys[2] = {y - ys[0], ys[1] - y};
        const float dxs[2] = {x - xs[0], xs[1] - x};
        for (int i = 0; i < channels; i++) {
            warped_image[i][j][k] = 0;
            for (int yi = 0; yi < 2; yi++) for (int xi = 0; xi < 2; xi++) {
                const float val = (ys[yi] < 0 || height-1 < ys[yi] || xs[xi] < 0 || width-1 < xs[xi]) ? 0 : image[i][ys[yi]][xs[xi]];
                warped_image[i][j][k] += dys[1-yi] * dxs[1-xi] * val;
            }
        }
    }
}