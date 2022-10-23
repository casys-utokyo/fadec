#include "config_7scenes.h"
#include "functional.h"
#include "activation.h"
#include "conv.h"
#include "layers.h"
#include "mnasnet.h"
#include "model.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
using namespace Eigen;

#include <dirent.h>
#include <sys/stat.h> // for mkdir

int conv_cnt;
int bn_cnt;
int add_cnt;
int other_cnt;
int act_cnt;

qwint* weights = new qwint[n_weights];
int w_idx[n_convs];
int w_shifts[n_convs];

qbint* biases = new qbint[n_biases];
int b_idx[n_convs];
int b_shifts[n_convs];

qsint* scales = new qsint[n_scales];
int s_idx[n_bns];
int s_shifts[n_bns];

int cin_shifts[n_convs];
int cout_shifts[n_convs];
int ain1_shifts[n_adds];
int ain2_shifts[n_adds];
int aout_shifts[n_adds];
int oin_shifts[n_others];
int oout_shifts[n_others];

int ln_cnt;


const string save_dir = "./results/";

void set_idx(string filename, const int n_files, int* start_idx) {
    int n_params[n_files];
    string filepath = param_folder + filename;
    ifstream ifs(filepath);
    if (!ifs) {
        cerr << "FileNotFound: " + filepath << "\n";
        exit(1);
    }
    ifs.read((char*) n_params, sizeof(int) * n_files);
    ifs.close();

    start_idx[0] = 0;
    for (int i = 0; i < n_files - 1; i++)
        start_idx[i+1] = start_idx[i] + n_params[i];
}


template<class T>
void set_param(string filename, const int n_params, T* params) {
    string filepath = param_folder + filename;
    ifstream ifs(filepath);
    if (!ifs) {
        cerr << "FileNotFound: " + filepath << "\n";
        exit(1);
    }
    ifs.read((char*) params, sizeof(T) * n_params);
    ifs.close();
}


void read_params() {
    set_idx("n_weights", n_convs, w_idx);
    set_param<qwint>("weights_quantized", n_weights, weights);
    set_param<int>("weight_shifts", n_convs, w_shifts);

    set_idx("n_biases", n_convs, b_idx);
    set_param<qbint>("biases_quantized", n_biases, biases);
    set_param<int>("bias_shifts", n_convs, b_shifts);

    set_idx("n_scales", n_bns, s_idx);
    set_param<qsint>("scales_quantized", n_scales, scales);
    set_param<int>("scale_shifts", n_bns, s_shifts);

    set_param<int>("cin_shifts", n_convs, cin_shifts);
    set_param<int>("cout_shifts", n_convs, cout_shifts);
    set_param<int>("ain1_shifts", n_adds, ain1_shifts);
    set_param<int>("ain2_shifts", n_adds, ain2_shifts);
    set_param<int>("aout_shifts", n_adds, aout_shifts);
    set_param<int>("oin_shifts", n_others, oin_shifts);
    set_param<int>("oout_shifts", n_others, oout_shifts);

    constexpr int irregulars[4] = {64, 72, 78, 93}; // 応急処置
    for (int idx : irregulars) cin_shifts[idx]--;
}


void predict(const qaint reference_image[3 * test_image_height * test_image_width],
             const int n_measurement_frames,
             const qaint measurement_feature_halfs[test_n_measurement_frames * fpn_output_channels * height_2 * width_2],
             const float* warpings,
             qaint reference_feature_half[fpn_output_channels * height_2 * width_2],
             qaint hidden_state[hid_channels * height_32 * width_32],
             qaint cell_state[hid_channels * height_32 * width_32],
             qaint depth_full[test_image_height * test_image_width],
             const string filename) {

    conv_cnt = 0;
    bn_cnt = 0;
    add_cnt = 0;
    other_cnt = 0;
    act_cnt = 0;
    ln_cnt = 0;

    const int act_in = act_cnt++;

    qaint layer1[channels_1 * height_2 * width_2];
    qaint layer2[channels_2 * height_4 * width_4];
    qaint layer3[channels_3 * height_8 * width_8];
    qaint layer4[channels_4 * height_16 * width_16];
    qaint layer5[channels_5 * height_32 * width_32];
    int act_out_layer1;
    int act_out_layer2;
    int act_out_layer3;
    int act_out_layer4;
    int act_out_layer5;
    FeatureExtractor(reference_image, layer1, layer2, layer3, layer4, layer5,
                     act_in, act_out_layer1, act_out_layer2, act_out_layer3, act_out_layer4, act_out_layer5);

    qaint reference_feature_quarter[fpn_output_channels * height_4 * width_4];
    qaint reference_feature_one_eight[fpn_output_channels * height_8 * width_8];
    qaint reference_feature_one_sixteen[fpn_output_channels * height_16 * width_16];
    int act_out_half;
    int act_out_quarter;
    int act_out_one_eight;
    int act_out_one_sixteen;
    FeatureShrinker(layer1, layer2, layer3, layer4, layer5, reference_feature_half, reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen,
                    act_out_layer1, act_out_layer2, act_out_layer3, act_out_layer4, act_out_layer5,
                    act_out_half, act_out_quarter, act_out_one_eight, act_out_one_sixteen);

    if (n_measurement_frames == 0) return;

    qaint cost_volume[n_depth_levels * height_2 * width_2];
    int act_out_cost_volume;
    cost_volume_fusion(reference_feature_half, n_measurement_frames, measurement_feature_halfs, warpings, cost_volume,
                       act_out_half, act_out_cost_volume);

    qaint skip0[hyper_channels * height_2 * width_2];
    qaint skip1[(hyper_channels * 2) * height_4 * width_4];
    qaint skip2[(hyper_channels * 4) * height_8 * width_8];
    qaint skip3[(hyper_channels * 8) * height_16 * width_16];
    qaint bottom[(hyper_channels * 16) * height_32 * width_32];
    int act_out_skip0;
    int act_out_skip1;
    int act_out_skip2;
    int act_out_skip3;
    int act_out_bottom;
    CostVolumeEncoder(reference_feature_half, reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen, cost_volume,
                      skip0, skip1, skip2, skip3, bottom, filename,
                      act_out_half, act_out_quarter, act_out_one_eight, act_out_one_sixteen, act_out_cost_volume,
                      act_out_skip0, act_out_skip1, act_out_skip2, act_out_skip3, act_out_bottom);

    int act_out_hidden_state;
    int act_out_cell_state;
    LSTMFusion(bottom, hidden_state, cell_state, filename, act_out_bottom, act_out_hidden_state, act_out_cell_state);

    int act_out_depth_full;
    CostVolumeDecoder(reference_image, skip0, skip1, skip2, skip3, hidden_state, depth_full,
                      act_in, act_out_skip0, act_out_skip1, act_out_skip2, act_out_skip3, act_out_hidden_state, act_out_depth_full);
}


int main() {
    printf("Predicting with System: %s\n", system_name.c_str());
    printf("# of Measurement Frames: %d\n", test_n_measurement_frames);

    const string test_dataset_names[8] = {"chess-seq-01", "chess-seq-02", "fire-seq-01", "fire-seq-02", "office-seq-01", "office-seq-03", "redkitchen-seq-01", "redkitchen-seq-07"};

    for (auto test_dataset_name : test_dataset_names) {
        float warp_grid[3][width_2 * height_2];
        get_warp_grid_for_cost_volume_calculation(warp_grid);

        printf("Predicting for scene:%s\n", scene.c_str());

        KeyframeBuffer keyframe_buffer;

        ifstream ifs;
        string file_buf;

        DIR *dir; struct dirent *diread;
        vector<string> files;

        if ((dir = opendir((scene_folder + test_dataset_name + "/").c_str())) != nullptr) {
            while ((diread = readdir(dir)) != nullptr) {
                if (diread->d_name[0] == '0')
                    files.push_back(diread->d_name);
            }
            closedir (dir);
        } else {
            cout << "FolderNotFound: " << test_dataset_name << "\n";
            continue;
        }
        sort(files.begin(), files.end());

        const int n_images = files.size();

        const string image_filedir = scene_folder + test_dataset_name + '/';
        const int len_image_filedir = image_filedir.length();
        string image_filenames[n_images];
        for (int i = 0; i < n_images; i++) {
            image_filenames[i] = image_filedir + files[i];
        }
        print1(image_filenames[0]);


        ifs.open(scene_folder + test_dataset_name + "/K.txt");
        if (!ifs) {
            cerr << "FileNotFound: " + scene_folder + test_dataset_name + "/K.txt" << "\n";
            exit(1);
        }
        float K[3][3];
        for (int i = 0; i < 3; i++) {
            getline(ifs, file_buf);
            istringstream iss(file_buf);
            string tmp;
            for (int j = 0; j < 3; j++) {
                iss >> tmp;
                K[i][j] = stof(tmp);
            }
        }
        ifs.close();

        float full_K[3][3];
        get_updated_intrinsics(K, full_K);

        float half_K[3][3];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) half_K[i][j] = full_K[i][j] / 2.0;
        for (int j = 0; j < 3; j++) half_K[2][j] = full_K[2][j];

        float lstm_K_bottom[3][3];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) lstm_K_bottom[i][j] = full_K[i][j] / 32.0;
        for (int j = 0; j < 3; j++) lstm_K_bottom[2][j] = full_K[2][j];

        ifs.open(scene_folder + test_dataset_name + "/poses.txt");
        if (!ifs) {
            cerr << "FileNotFound: " + scene_folder + test_dataset_name + "/poses.txt" << "\n";
            exit(1);
        }
        vector<float> tmp_poses;
        while (getline(ifs, file_buf)) {
            istringstream iss(file_buf);
            string tmp;
            for (int i = 0; i < 16; i++) {
                iss >> tmp;
                tmp_poses.push_back(stof(tmp));
            }
        }
        ifs.close();

        float poses[n_images][4 * 4];
        int poses_idx = 0;
        for (int i = 0; i < n_images; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    poses[i][j * 4 + k] = tmp_poses[poses_idx];
                    poses_idx++;
                }
            }
        }

        // read params
        read_params();

        bool previous_exists = false;
        float previous_depth[test_image_height][test_image_width];
        float previous_pose[4 * 4];

        bool state_exists = false;
        qaint hidden_state[hid_channels * height_32 * width_32];
        qaint cell_state[hid_channels * height_32 * width_32];

        ofstream ofs;
        double min_time = 10000;
        double max_time = 0;
        double mean_time = 0;
        int loops = 0;
        for (int f = 0; f < n_images; f++) {
            clock_t start = clock();

            float reference_pose[4 * 4];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) reference_pose[i * 4 + j] = poses[f][i * 4 + j];

            // POLL THE KEYFRAME BUFFER
            const int response = keyframe_buffer.try_new_keyframe(reference_pose);
            cout << image_filenames[f].substr(len_image_filedir) << ": " << response << "\n";

            if (response == 2 || response == 4 || response == 5) continue;
            else if (response == 3) {
                previous_exists = false;
                state_exists = false;
                continue;
            }

            float reference_image_float[3 * test_image_height * test_image_width];
            load_image(image_filenames[f], reference_image_float);
            qaint reference_image[3 * test_image_height * test_image_width];
            const int ashift = cin_shifts[0];
            for (int idx = 0; idx < 3 * test_image_height * test_image_width; idx++)
                reference_image[idx] = reference_image_float[idx] * (1 << ashift);

            float measurement_poses[test_n_measurement_frames * 4 * 4];
            qaint measurement_feature_halfs[test_n_measurement_frames * fpn_output_channels * height_2 * width_2];
            const int n_measurement_frames = keyframe_buffer.get_best_measurement_frames(reference_pose, measurement_poses, measurement_feature_halfs);

            // prepare for cost volume fusion
            float* warpings = new float[n_measurement_frames * n_depth_levels * height_2 * width_2 * 2];

            for (int m = 0; m < n_measurement_frames; m++) {
                Matrix4f pose1, pose2;
                for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) pose1(i, j) = reference_pose[i * 4 + j];
                for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) pose2(i, j) = measurement_poses[(m * 4 + i) * 4 + j];

                Matrix4f extrinsic2 = pose2.inverse() * pose1;
                Matrix3f R = extrinsic2.block(0, 0, 3, 3);
                Vector3f t = extrinsic2.block(0, 3, 3, 1);

                Matrix3f K;
                for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) K(i, j) = half_K[i][j];
                MatrixXf wg(3, width_2 * height_2);
                for (int i = 0; i < 3; i++) for (int j = 0; j < width_2 * height_2; j++) wg(i, j) = warp_grid[i][j];

                Vector3f _Kt = K * t;
                Matrix3f K_R_Kinv = K * R * K.inverse();
                MatrixXf K_R_Kinv_UV(3, width_2 * height_2);
                K_R_Kinv_UV = K_R_Kinv * wg;

                MatrixXf Kt(3, width_2 * height_2);
                for (int i = 0; i < width_2 * height_2; i++) Kt.block(0, i, 3, 1) = _Kt;

                for (int depth_i = 0; depth_i < n_depth_levels; depth_i++) {
                    const float this_depth = 1.0 / (inverse_depth_base + depth_i * inverse_depth_step);

                    MatrixXf _warping(width_2 * height_2, 3);
                    _warping = (K_R_Kinv_UV + (Kt / this_depth)).transpose();

                    MatrixXf _warping0(width_2 * height_2, 2);
                    VectorXf _warping1(width_2 * height_2);
                    _warping0 = _warping.block(0, 0, width_2 * height_2, 2);
                    _warping1 = _warping.block(0, 2, width_2 * height_2, 1).array() + 1e-8f;

                    _warping0.block(0, 0, width_2 * height_2, 1).array() /= _warping1.array();
                    _warping0.block(0, 0, width_2 * height_2, 1).array() -= width_normalizer;
                    _warping0.block(0, 0, width_2 * height_2, 1).array() /= width_normalizer;

                    _warping0.block(0, 1, width_2 * height_2, 1).array() /= _warping1.array();
                    _warping0.block(0, 1, width_2 * height_2, 1).array() -= height_normalizer;
                    _warping0.block(0, 1, width_2 * height_2, 1).array() /= height_normalizer;

                    for (int idx = 0; idx < height_2 * width_2; idx++) for (int k = 0; k < 2; k++)
                        warpings[((m * n_depth_levels + depth_i) * (height_2 * width_2) + idx) * 2 + k] = _warping0(idx, k);
                }
            }

            // prepare depth_estimation
            float depth_estimation[1][height_32][width_32];
            if (previous_exists) {
                float depth_hypothesis[1][height_2][width_2];
                get_non_differentiable_rectangle_depth_estimation(reference_pose, previous_pose, previous_depth,
                                                                    full_K, half_K,
                                                                    depth_hypothesis);
                interpolate<1, height_2, width_2, height_32, width_32>(depth_hypothesis, depth_estimation);
            } else {
                for (int i = 0 ; i < height_32; i++) for (int j = 0; j < width_32; j++)
                    depth_estimation[0][i][j] = 0;
            }

            // initialize ConvLSTM params if needed
            if (!state_exists) {
                for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
                    hidden_state[idx] = 0;
                for (int idx = 0; idx < hid_channels * height_32 * width_32; idx++)
                    cell_state[idx] = 0;
            }

            if (previous_exists) {
                Matrix4f p_pose, c_pose;
                for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) p_pose(i, j) = previous_pose[i * 4 + j];
                for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) c_pose(i, j) = reference_pose[i * 4 + j];

                Matrix4f transformation = p_pose.inverse() * c_pose;
                float trans[4][4];
                for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) trans[i][j] = transformation(i, j);

                float in_hidden_state[hid_channels][height_32][width_32];
                for (int i = 0; i < hid_channels; i++) for (int j = 0; j < height_32; j++) for (int k = 0; k < width_32; k++)
                    in_hidden_state[i][j][k] = hidden_state[(i * height_32 + j) * width_32 + k] / (float) (1 << hiddenshift);
                float out_hidden_state[hid_channels][height_32][width_32];
                warp_frame_depth(in_hidden_state, depth_estimation[0], trans, lstm_K_bottom, out_hidden_state);

                for (int i = 0; i < hid_channels; i++) for (int j = 0; j < height_32; j++) for (int k = 0; k < width_32; k++)
                    hidden_state[(i * height_32 + j) * width_32 + k] = (depth_estimation[0][j][k] <= 0.01) ? 0.0 : out_hidden_state[i][j][k] * (1 << hiddenshift);
            }

            qaint reference_feature_half[fpn_output_channels * height_2 * width_2];
            qaint depth_full[test_image_height * test_image_width];
            predict(reference_image, n_measurement_frames, measurement_feature_halfs,
                    warpings, reference_feature_half, hidden_state, cell_state, depth_full, image_filenames[f].substr(len_image_filedir, 5));
            delete[] warpings;

            keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half);
            if (response == 0) continue;

            float prediction[test_image_height * test_image_width];
            for (int idx = 0; idx < test_image_height * test_image_width; idx++)
                prediction[idx] = 1.0 / (inverse_depth_multiplier * (depth_full[idx] / (float) (1 << sigshift)) + inverse_depth_base);

            for (int i = 0 ; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++)
                previous_depth[i][j] = prediction[i * test_image_width + j];
            for (int i = 0 ; i < 4; i++) for (int j = 0; j < 4; j++)
                previous_pose[i * 4 + j] = reference_pose[i * 4 + j];
            previous_exists = true;

            state_exists = true;

            clock_t end = clock();
            double time_cur = (double)(end - start) / CLOCKS_PER_SEC;
            cout << time_cur << " [s]\n";
            min_time = min(min_time, time_cur);
            max_time = max(max_time, time_cur);
            mean_time += time_cur;
            loops++;

            mkdir(("./results_7scenes/" + test_dataset_name).c_str(), 0775);

            string output_filepath = "./results_7scenes/" + test_dataset_name + '/' + image_filenames[f].substr(len_image_filedir, 6) + ".bin";
            ofs.open(output_filepath, ios::out|ios::binary|ios::trunc);
            if (!ofs) {
                cerr << "FileNotFound: " + output_filepath << "\n";
                exit(1);
            }
            for (int i = 0 ; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++)
                ofs.write((char*) &previous_depth[i][j], sizeof(float));
            ofs.close();
        }

        keyframe_buffer.close();

        print2("loops    :", loops);
        print2("Min  time:", min_time);
        print2("Max  time:", max_time);
        print2("Mean time:", mean_time / loops);
    }

    delete[] weights;
    delete[] biases;
    delete[] scales;

    return 0;
}
