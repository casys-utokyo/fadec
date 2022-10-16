#pragma once
#include "settings.h"

constexpr int org_image_width = 540;
constexpr int org_image_height = 360;

constexpr int test_image_width = 96;
constexpr int test_image_height = 64;

constexpr int test_n_measurement_frames = 2;
constexpr int test_keyframe_buffer_size = 30;
constexpr float test_keyframe_pose_distance = 0.1;
constexpr float test_optimal_t_measure = 0.15;
constexpr float test_optimal_R_measure = 0.0;

// SET THESE: TESTING FOLDER LOCATIONS
const string dataset_name = "hololens-dataset";
const string system_name = "keyframe_" + dataset_name + "_" + to_string(test_image_width) + "_" + to_string(test_image_height) + "_" + to_string(test_n_measurement_frames) + "_dvmvs_fusionnet_online";

const string scene_folder = "./params_cpp";
const string scene = "000";

constexpr int n_test_frames = 50;

constexpr float scale_rgb = 255.0;
constexpr float mean_rgb[3] = {0.485, 0.456, 0.406};
constexpr float std_rgb[3] = {0.229, 0.224, 0.225};

constexpr float min_depth = 0.25;
constexpr float max_depth = 20.0;
constexpr int n_depth_levels = 64;

constexpr float inverse_depth_base = 1.0 / max_depth;
constexpr float inverse_depth_multiplier = 1.0 / min_depth - 1.0 / max_depth;
constexpr float inverse_depth_step = inverse_depth_multiplier / (n_depth_levels - 1);

constexpr int fpn_output_channels = 32;
constexpr int hyper_channels = 32;
constexpr int hid_channels = hyper_channels * 16;

constexpr int channels_1 = 16;
constexpr int channels_2 = 24;
constexpr int channels_3 = 40;
constexpr int channels_4 = 96;
constexpr int channels_5 = 320;

constexpr int height_2 = test_image_height / 2;
constexpr int height_4 = test_image_height / 4;
constexpr int height_8 = test_image_height / 8;
constexpr int height_16 = test_image_height / 16;
constexpr int height_32 = test_image_height / 32;

constexpr int width_2 = test_image_width / 2;
constexpr int width_4 = test_image_width / 4;
constexpr int width_8 = test_image_width / 8;
constexpr int width_16 = test_image_width / 16;
constexpr int width_32 = test_image_width / 32;

constexpr float height_normalizer = height_2 / 2.0;
constexpr float width_normalizer = width_2 / 2.0;

extern float* params;
constexpr int n_files = 255 + 18 + 80 + 1 + 80;
extern int start_idx[n_files + 1];
extern int param_cnt;


#define conv_out_size(size, kernel_size, stride, padding) ((size) + 2 * (padding) - (kernel_size)) / (stride) + 1
#define invres_out_size(size, kernel_size, stride) conv_out_size((size), (kernel_size), (stride), (kernel_size) / 2)
#define stack_out_size(size, kernel_size, stride) invres_out_size((size), (kernel_size), (stride))

#define new_2d(arr, d0, d1) for (int iii2 = 0; iii2 < (d0); iii2++) {(arr)[iii2] = new float[(d1)];}
#define new_3d(arr, d0, d1, d2) for (int iii3 = 0; iii3 < (d0); iii3++) {(arr)[iii3] = new float*[(d1)]; new_2d((arr)[iii3], (d1), (d2));}
#define new_4d(arr, d0, d1, d2, d3) for (int iii4 = 0; iii4 < (d0); iii4++) {(arr)[iii4] = new float**[(d1)]; new_3d((arr)[iii4], (d1), (d2), (d3));}

#define tmp_delete_2d(arr, d0, d1) for (int iii2 = 0; iii2 < (d0); iii2++) {delete[] (arr)[iii2];}
#define tmp_delete_3d(arr, d0, d1, d2) for (int iii3 = 0; iii3 < (d0); iii3++) {tmp_delete_2d((arr)[iii3], (d1), (d2)); delete[] (arr)[iii3];}
#define tmp_delete_4d(arr, d0, d1, d2, d3) for (int iii4 = 0; iii4 < (d0); iii4++) {tmp_delete_3d((arr)[iii4], (d1), (d2), (d3)); delete[] (arr)[iii4];}

#define delete_2d(arr, d0, d1) tmp_delete_2d(arr, d0, d1) ; delete[] (arr);
#define delete_3d(arr, d0, d1, d2) tmp_delete_3d(arr, d0, d1, d2) ; delete[] (arr);
#define delete_4d(arr, d0, d1, d2, d3) tmp_delete_4d(arr, d0, d1, d2, d3); delete[] (arr);


// utils
void pose_distance(const float reference_pose[4 * 4], const float measurement_pose[4 * 4], float &combined_measure, float &R_measure, float &t_measure);
void get_warp_grid_for_cost_volume_calculation(float warp_grid[3][width_2 * height_2]);
void cost_volume_fusion(const float image1[fpn_output_channels * height_2 * width_2],
                        const int n_measurement_frames,
                        const float image2s[test_n_measurement_frames * fpn_output_channels * height_2 * width_2],
                        const float* warpings,
                        float fused_cost_volume[n_depth_levels * height_2 * width_2]);
void get_non_differentiable_rectangle_depth_estimation(const float reference_pose[4 * 4],
                                                       const float measurement_pose[4 * 4],
                                                       const float previous_depth[test_image_height][test_image_width],
                                                       const float full_K[3][3],
                                                       const float half_K[3][3],
                                                       float depth_hypothesis[1][height_2][width_2]);
void warp_frame_depth(const float image_src[hyper_channels * 16][height_32][width_32],
                      const float depth_dst[height_32][width_32],
                      const float trans[4][4],
                      const float camera_matrix[3][3],
                      float image_dst[hyper_channels * 16][height_32][width_32]);
bool is_pose_available(const float pose[4 * 4]);

// keyframe_buffer
class KeyframeBuffer{
public:
    KeyframeBuffer(){
        new_2d(buffer_poses, buffer_size, 4 * 4);
        new_2d(buffer_feature_halfs, buffer_size, fpn_output_channels * height_2 * width_2);
    }

    int try_new_keyframe(const float pose[4 * 4]);
    void add_new_keyframe(const float pose[4 * 4], const float feature_half[fpn_output_channels * height_2 * width_2]);
    int get_best_measurement_frames(const float reference_pose[4 * 4], float measurement_poses[test_n_measurement_frames * 4 * 4], float measurement_feature_halfs[test_keyframe_buffer_size * fpn_output_channels * height_2 * width_2]);

    void close() {
        delete_2d(buffer_poses, buffer_size, 4 * 4);
        delete_2d(buffer_feature_halfs, buffer_size, fpn_output_channels * height_2 * width_2);
    }

private:
    const int buffer_size = test_keyframe_buffer_size;
    int buffer_idx = 0;
    int buffer_cnt = 0;
    float **buffer_poses = new float*[test_keyframe_buffer_size];
    float **buffer_feature_halfs = new float*[test_keyframe_buffer_size];
    const float optimal_R_score = test_optimal_R_measure;
    const float optimal_t_score = test_optimal_t_measure;
    const float keyframe_pose_distance = test_keyframe_pose_distance;
    int __tracking_lost_counter = 0;
    float calculate_penalty(const float t_score, const float R_score);
};

// dataset_loader
void get_updated_intrinsics(const float K[3][3], float updated_intrinsic[3][3]);
void load_image(const string image_filename, float reference_image[3 * test_image_height * test_image_width]);
void save_image(const string image_filename, float depth[test_image_height][test_image_width]);
