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

const string param_folder = "./params_cpp_with_ptq";
const string scene_folder = "./images";
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

constexpr int n_convs = 96;
constexpr int n_bns = 81;
constexpr int n_adds = 14;
constexpr int n_others = 77;

extern int conv_cnt;
extern int bn_cnt;
extern int add_cnt;
extern int other_cnt;
extern int act_cnt;

typedef char qwint;
constexpr int n_weights = 34619560;
extern qwint* weights;
extern int w_idx[n_convs];
extern int w_shifts[n_convs];

typedef int qbint;
constexpr int n_biases = 24885;
extern qbint* biases;
extern int b_idx[n_convs];
extern int b_shifts[n_convs];

typedef char qsint;
constexpr int n_scales = 22544;
extern qsint* scales;
extern int s_idx[n_bns];
extern int s_shifts[n_bns];

typedef short qaint;
extern int cin_shifts[n_convs];
extern int cout_shifts[n_convs];
extern int ain1_shifts[n_adds];
extern int ain2_shifts[n_adds];
extern int aout_shifts[n_adds];
extern int oin_shifts[n_others];
extern int oout_shifts[n_others];

typedef int qmint;
constexpr qmint QA_MIN = -32768;
constexpr qmint QA_MAX = 32767;

constexpr int cellshift = 12;
constexpr int hiddenshift = 14;

// layer_norm
extern int ln_cnt;
constexpr int lnout_shifts[2] = {12, 12};

// 8 bit -> 12, 14 bit
typedef unsigned char qtint;
constexpr int tbbit = 8;
constexpr int tbshift = 5;
constexpr int celushift = 12;
constexpr int sigshift = 14;
constexpr qaint celu_table[256] = {0, -126, -248, -367, -481, -593, -700, -805, -906, -1004, -1099, -1192, -1281, -1367, -1451, -1533, -1612, -1688, -1762, -1834, -1904, -1971, -2036, -2100, -2161, -2221, -2278, -2334, -2389, -2441, -2492, -2541, -2589, -2636, -2680, -2724, -2766, -2807, -2847, -2885, -2922, -2959, -2994, -3027, -3060, -3092, -3123, -3153, -3182, -3210, -3237, -3264, -3289, -3314, -3338, -3362, -3384, -3406, -3427, -3448, -3468, -3487, -3506, -3524, -3542, -3559, -3575, -3591, -3607, -3622, -3636, -3651, -3664, -3678, -3690, -3703, -3715, -3727, -3738, -3749, -3760, -3770, -3780, -3790, -3799, -3808, -3817, -3826, -3834, -3842, -3850, -3858, -3865, -3872, -3879, -3886, -3892, -3898, -3904, -3910, -3916, -3922, -3927, -3932, -3937, -3942, -3947, -3951, -3956, -3960, -3964, -3968, -3972, -3976, -3980, -3983, -3987, -3990, -3993, -3997, -4000, -4003, -4006, -4008, -4011, -4014, -4016, -4019, -4021, -4023, -4026, -4028, -4030, -4032, -4034, -4036, -4038, -4039, -4041, -4043, -4044, -4046, -4048, -4049, -4050, -4052, -4053, -4055, -4056, -4057, -4058, -4059, -4061, -4062, -4063, -4064, -4065, -4066, -4067, -4068, -4068, -4069, -4070, -4071, -4072, -4072, -4073, -4074, -4075, -4075, -4076, -4076, -4077, -4078, -4078, -4079, -4079, -4080, -4080, -4081, -4081, -4082, -4082, -4083, -4083, -4083, -4084, -4084, -4084, -4085, -4085, -4086, -4086, -4086, -4086, -4087, -4087, -4087, -4088, -4088, -4088, -4088, -4089, -4089, -4089, -4089, -4089, -4090, -4090, -4090, -4090, -4090, -4091, -4091, -4091, -4091, -4091, -4091, -4091, -4092, -4092, -4092, -4092, -4092, -4092, -4092, -4092, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4095, -4095};
constexpr qaint Sigmoid_table[256] = {8192, 8320, 8448, 8576, 8703, 8831, 8958, 9084, 9211, 9336, 9462, 9586, 9710, 9833, 9956, 10078, 10198, 10318, 10437, 10555, 10672, 10788, 10902, 11015, 11128, 11239, 11348, 11457, 11564, 11669, 11773, 11876, 11978, 12078, 12176, 12273, 12369, 12463, 12555, 12646, 12735, 12823, 12909, 12994, 13077, 13159, 13239, 13318, 13395, 13471, 13545, 13617, 13689, 13758, 13826, 13893, 13958, 14022, 14085, 14146, 14206, 14264, 14321, 14377, 14431, 14484, 14536, 14587, 14636, 14684, 14731, 14777, 14822, 14865, 14908, 14949, 14990, 15029, 15067, 15105, 15141, 15177, 15211, 15245, 15277, 15309, 15340, 15370, 15400, 15428, 15456, 15483, 15509, 15535, 15559, 15584, 15607, 15630, 15652, 15673, 15694, 15715, 15735, 15754, 15772, 15791, 15808, 15825, 15842, 15858, 15874, 15889, 15904, 15918, 15932, 15946, 15959, 15971, 15984, 15996, 16008, 16019, 16030, 16041, 16051, 16061, 16071, 16080, 16089, 16098, 16107, 16115, 16123, 16131, 16139, 16146, 16154, 16161, 16167, 16174, 16180, 16187, 16193, 16198, 16204, 16209, 16215, 16220, 16225, 16230, 16234, 16239, 16243, 16248, 16252, 16256, 16260, 16264, 16267, 16271, 16274, 16278, 16281, 16284, 16287, 16290, 16293, 16296, 16298, 16301, 16304, 16306, 16308, 16311, 16313, 16315, 16317, 16319, 16321, 16323, 16325, 16327, 16329, 16330, 16332, 16334, 16335, 16337, 16338, 16340, 16341, 16342, 16343, 16345, 16346, 16347, 16348, 16349, 16350, 16351, 16352, 16353, 16354, 16355, 16356, 16357, 16358, 16359, 16359, 16360, 16361, 16362, 16362, 16363, 16364, 16364, 16365, 16365, 16366, 16367, 16367, 16368, 16368, 16369, 16369, 16370, 16370, 16370, 16371, 16371, 16372, 16372, 16372, 16373, 16373, 16373, 16374, 16374, 16374, 16375, 16375, 16375, 16375, 16376, 16376, 16376, 16376, 16377, 16377, 16377, 16377, 16378, 16378, 16378, 16378, 16378};

#define clip(v) min(max((qmint) round(v), QA_MIN), QA_MAX)

#define conv_out_size(size, kernel_size, stride, padding) ((size) + 2 * (padding) - (kernel_size)) / (stride) + 1
#define invres_out_size(size, kernel_size, stride) conv_out_size((size), (kernel_size), (stride), (kernel_size) / 2)
#define stack_out_size(size, kernel_size, stride) invres_out_size((size), (kernel_size), (stride))

#define new_2d_qaint(arr, d0, d1) for (int iii2 = 0; iii2 < (d0); iii2++) {(arr)[iii2] = new qaint[(d1)];}
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
void cost_volume_fusion(const qaint image1[fpn_output_channels * height_2 * width_2],
                        const int n_measurement_frames,
                        const qaint image2s[test_n_measurement_frames * fpn_output_channels * height_2 * width_2],
                        const float* warpings,
                        qaint fused_cost_volume[n_depth_levels * height_2 * width_2],
                        const int act_in, int& act_out);
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
        new_2d_qaint(buffer_feature_halfs, buffer_size, fpn_output_channels * height_2 * width_2);
    }

    int try_new_keyframe(const float pose[4 * 4]);
    void add_new_keyframe(const float pose[4 * 4], const qaint feature_half[fpn_output_channels * height_2 * width_2]);
    int get_best_measurement_frames(const float reference_pose[4 * 4], float measurement_poses[test_n_measurement_frames * 4 * 4], qaint measurement_feature_halfs[test_keyframe_buffer_size * fpn_output_channels * height_2 * width_2]);

    void close() {
        delete_2d(buffer_poses, buffer_size, 4 * 4);
        delete_2d(buffer_feature_halfs, buffer_size, fpn_output_channels * height_2 * width_2);
    }

private:
    const int buffer_size = test_keyframe_buffer_size;
    int buffer_idx = 0;
    int buffer_cnt = 0;
    float **buffer_poses = new float*[test_keyframe_buffer_size];
    qaint **buffer_feature_halfs = new qaint*[test_keyframe_buffer_size];
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
