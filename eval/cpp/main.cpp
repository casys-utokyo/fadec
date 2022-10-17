#include "config.h"
#include "functional.h"
#include "conv.h"
#include "activation.h"
#include "batchnorm.h"
#include "layers.h"
#include "mnasnet.h"
#include "model.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
using namespace Eigen;

float* params = new float[2725512 + 62272 + 8990848 + 18874368 + 4066277];
int start_idx[n_files + 1];
int param_cnt;

void read_params() {
    ifstream ifs;

    int n_params[n_files];
    ifs.open(param_folder + "values");
    if (!ifs) {
        cerr << "FileNotFound: " + param_folder + "values" << "\n";
        exit(1);
    }
    ifs.read((char*) n_params, sizeof(int) * n_files);
    ifs.close();

    start_idx[0] = 0;
    for (int i = 0; i < n_files; i++)
        start_idx[i+1] = start_idx[i] + n_params[i];

    ifs.open(param_folder + "params");
    if (!ifs) {
        cerr << "FileNotFound: " + param_folder + "params" << "\n";
        exit(1);
    }
    ifs.read((char*) params, sizeof(float) * start_idx[n_files]);
    ifs.close();
}


void predict(const float reference_image[3 * test_image_height * test_image_width],
             const int n_measurement_frames,
             const float measurement_feature_halfs[test_n_measurement_frames * fpn_output_channels * height_2 * width_2],
             const float* warpings,
             float reference_feature_half[fpn_output_channels * height_2 * width_2],
             float hidden_state[hid_channels * height_32 * width_32],
             float cell_state[hid_channels * height_32 * width_32],
             float prediction[test_image_height * test_image_width]) {

    param_cnt = 0;

    float layer1[channels_1 * height_2 * width_2];
    float layer2[channels_2 * height_4 * width_4];
    float layer3[channels_3 * height_8 * width_8];
    float layer4[channels_4 * height_16 * width_16];
    float layer5[channels_5 * height_32 * width_32];
    FeatureExtractor(reference_image, layer1, layer2, layer3, layer4, layer5);

    float reference_feature_quarter[fpn_output_channels * height_4 * width_4];
    float reference_feature_one_eight[fpn_output_channels * height_8 * width_8];
    float reference_feature_one_sixteen[fpn_output_channels * height_16 * width_16];
    FeatureShrinker(layer1, layer2, layer3, layer4, layer5, reference_feature_half, reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen);

    if (n_measurement_frames == 0) return;

    float cost_volume[n_depth_levels * height_2 * width_2];
    cost_volume_fusion(reference_feature_half, n_measurement_frames, measurement_feature_halfs, warpings, cost_volume);

    float skip0[hyper_channels * height_2 * width_2];
    float skip1[(hyper_channels * 2) * height_4 * width_4];
    float skip2[(hyper_channels * 4) * height_8 * width_8];
    float skip3[(hyper_channels * 8) * height_16 * width_16];
    float bottom[(hyper_channels * 16) * height_32 * width_32];
    CostVolumeEncoder(reference_feature_half, reference_feature_quarter, reference_feature_one_eight, reference_feature_one_sixteen, cost_volume,
                      skip0, skip1, skip2, skip3, bottom);

    LSTMFusion(bottom, hidden_state, cell_state);

    CostVolumeDecoder(reference_image, skip0, skip1, skip2, skip3, hidden_state, prediction);
}


int main() {
    printf("Predicting with System: %s\n", system_name.c_str());
    printf("# of Measurement Frames: %d\n", test_n_measurement_frames);

    float warp_grid[3][width_2 * height_2];
    get_warp_grid_for_cost_volume_calculation(warp_grid);

    printf("Predicting for scene:%s\n", scene.c_str());

    KeyframeBuffer keyframe_buffer;

    ifstream ifs;
    string file_buf;

    ifs.open(scene_folder + "K.txt");
    if (!ifs) {
        cerr << "FileNotFound: " + scene_folder + "K.txt" << "\n";
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

    ifs.open(scene_folder + "poses.txt");
    if (!ifs) {
        cerr << "FileNotFound: " + scene_folder + "poses.txt" << "\n";
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

    // const int n_poses = tmp_poses.size() / 16;
    float poses[n_test_frames][4 * 4];
    int poses_idx = 0;
    for (int i = 0; i < n_test_frames; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                poses[i][j * 4 + k] = tmp_poses[poses_idx];
                poses_idx++;
            }
        }
    }
    // print1(n_poses);

    const string image_filedir = scene_folder;
    const int len_image_filedir = image_filedir.length();
    string image_filenames[n_test_frames];
    for (int i = 0; i < n_test_frames; i++) {
        ostringstream sout;
        sout << setfill('0') << setw(5) << i+3;
        image_filenames[i] = image_filedir + sout.str();
    }
    print1(image_filenames[0]);

    // read params
    read_params();

    bool previous_exists = false;
    float previous_depth[test_image_height][test_image_width];
    float previous_pose[4 * 4];

    bool state_exists = false;
    float hidden_state[hid_channels * height_32 * width_32];
    float cell_state[hid_channels * height_32 * width_32];

    ofstream ofs;
    double min_time = 10000;
    double max_time = 0;
    double mean_time = 0;
    int loops = 0;
    for (int f = 0; f < n_test_frames; f++) {
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

        float reference_image[3 * test_image_height * test_image_width];
        load_image(image_filenames[f], reference_image);

        float measurement_poses[test_n_measurement_frames * 4 * 4];
        float measurement_feature_halfs[test_n_measurement_frames * fpn_output_channels * height_2 * width_2];
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
                in_hidden_state[i][j][k] = hidden_state[(i * height_32 + j) * width_32 + k];
            float out_hidden_state[hid_channels][height_32][width_32];
            warp_frame_depth(in_hidden_state, depth_estimation[0], trans, lstm_K_bottom, out_hidden_state);

            for (int i = 0; i < hid_channels; i++) for (int j = 0; j < height_32; j++) for (int k = 0; k < width_32; k++)
                hidden_state[(i * height_32 + j) * width_32 + k] = (depth_estimation[0][j][k] <= 0.01) ? 0.0 : out_hidden_state[i][j][k];
        }

        float reference_feature_half[fpn_output_channels * height_2 * width_2];
        float prediction[test_image_height * test_image_width];
        predict(reference_image, n_measurement_frames, measurement_feature_halfs,
                warpings, reference_feature_half, hidden_state, cell_state, prediction);
        delete[] warpings;

        keyframe_buffer.add_new_keyframe(reference_pose, reference_feature_half);
        if (response == 0) continue;

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

        string output_filepath = "./results/" + image_filenames[f].substr(len_image_filedir, 5) + ".bin";
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

    delete[] params;
    return 0;
}
