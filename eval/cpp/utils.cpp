#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
using namespace Eigen;

#include "config.h"
#include "functional.h"
#include "kornia.h"

// GEOMETRIC UTILS
void pose_distance(const float reference_pose[4 * 4], const float measurement_pose[4 * 4], float &combined_measure, float &R_measure, float &t_measure) {
    // :param reference_pose: 4x4 numpy array, reference frame camera-to-world pose (not extrinsic matrix!)
    // :param measurement_pose: 4x4 numpy array, measurement frame camera-to-world pose (not extrinsic matrix!)
    // :return combined_measure: float, combined pose distance measure
    // :return R_measure: float, rotation distance measure
    // :return t_measure: float, translation distance measure

    Matrix4f r_pose, m_pose;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) r_pose(i, j) = reference_pose[i * 4 + j];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) m_pose(i, j) = measurement_pose[i * 4 + j];

    Matrix4f rel_pose = r_pose.inverse() * m_pose;
    Matrix3f R = rel_pose.block(0, 0, 3, 3);
    Vector3f t = rel_pose.block(0, 3, 3, 1);

    R_measure = sqrt(2 * (1 - min((float) 3.0, (float) R.trace()) / 3));
    t_measure = t.norm();
    combined_measure = sqrt(t_measure * t_measure + R_measure * R_measure);
}


void get_warp_grid_for_cost_volume_calculation(float warp_grid[3][width_2 * height_2]) {
    for (int i = 0; i < height_2; i++) for (int j = 0; j < width_2; j++) warp_grid[0][width_2 * i + j] = j;
    for (int i = 0; i < height_2; i++) for (int j = 0; j < width_2; j++) warp_grid[1][width_2 * i + j] = i;
    for (int i = 0; i < height_2; i++) for (int j = 0; j < width_2; j++) warp_grid[2][width_2 * i + j] = 1;
}


void calculate_cost_volume_by_warping(const float image1[fpn_output_channels * height_2 * width_2],
                                      const float* image2,
                                      const float* warping,
                                      float cost_volume[n_depth_levels * height_2 * width_2]) {

    for (int idx = 0; idx < n_depth_levels * height_2 * width_2; idx++)
        cost_volume[idx] = 0;

    for (int depth_i = 0; depth_i < n_depth_levels; depth_i++) {
        float warped_image2[fpn_output_channels * height_2 * width_2];
        grid_sample(image2, warping + depth_i * (height_2 * width_2 * 2), warped_image2, fpn_output_channels, height_2, width_2);

        for (int i = 0; i < fpn_output_channels; i++) for (int idx = 0; idx < height_2 * width_2; idx++)
            cost_volume[depth_i * (height_2 * width_2) + idx] += (image1[i * (height_2 * width_2) + idx] * warped_image2[i * (height_2 * width_2) + idx]) / fpn_output_channels;

    }
}


void cost_volume_fusion(const float image1[fpn_output_channels * height_2 * width_2],
                        const int n_measurement_frames,
                        const float image2s[test_n_measurement_frames * fpn_output_channels * height_2 * width_2],
                        const float* warpings,
                        float fused_cost_volume[n_depth_levels * height_2 * width_2]) {

    for (int idx = 0; idx < n_depth_levels * height_2 * width_2; idx++)
        fused_cost_volume[idx] = 0;

    for (int m = 0; m < n_measurement_frames; m++) {
        float cost_volume[n_depth_levels * height_2 * width_2];
        const float* image2 = image2s + m * (fpn_output_channels * height_2 * width_2);
        const float* warping = warpings + m * (n_depth_levels * height_2 * width_2 * 2);
        calculate_cost_volume_by_warping(image1, image2, warping, cost_volume);
        for (int idx = 0; idx < n_depth_levels * height_2 * width_2; idx++)
            fused_cost_volume[idx] += cost_volume[idx];
    }

    for (int idx = 0; idx < n_depth_levels * height_2 * width_2; idx++)
        fused_cost_volume[idx] /= n_measurement_frames;

}


void get_non_differentiable_rectangle_depth_estimation(const float reference_pose[4 * 4],
                                                       const float measurement_pose[4 * 4],
                                                       const float previous_depth[test_image_height][test_image_width],
                                                       const float full_K[3][3],
                                                       const float half_K[3][3],
                                                       float depth_hypothesis[1][height_2][width_2]) {

    Matrix4f r_pose, m_pose;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) r_pose(i, j) = reference_pose[i * 4 + j];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) m_pose(i, j) = measurement_pose[i * 4 + j];

    Matrix4f trans = r_pose.inverse() * m_pose;

    float points_3d_src[test_image_height][test_image_width][3];
    depth_to_3d<test_image_height, test_image_width>(previous_depth, full_K, points_3d_src);
    float points_3d_dst[test_image_height][test_image_width][3];
    transform_points<test_image_height, test_image_width>(trans, points_3d_src, points_3d_dst);

    pair<float, pair<int, int>> org_z_values[test_image_height * test_image_width];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) {
        const float z_value = max(0.0f, points_3d_dst[i][j][2]);
        org_z_values[test_image_width * i + j] = pair<float, pair<int, int>>(z_value, pair<float, float>(i, j));
    }

    sort(org_z_values, org_z_values + (test_image_height * test_image_width), greater<pair<float, pair<int, int>>>());

    float z_values[test_image_height][test_image_width];
    int sorting_indices[test_image_height][test_image_width][2];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) {
        const int idx = test_image_width * i + j;
        z_values[i][j] = org_z_values[idx].first;
        sorting_indices[i][j][0] = org_z_values[idx].second.first;
        sorting_indices[i][j][1] = org_z_values[idx].second.second;
    }

    float points_3d_sorted[test_image_height][test_image_width][3];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) for (int k = 0; k < 3; k++)
        points_3d_sorted[i][j][k] = points_3d_dst[sorting_indices[i][j][0]][sorting_indices[i][j][1]][k];

    float projections_float[test_image_height][test_image_width][2];
    project_points<test_image_height, test_image_width>(points_3d_sorted, half_K, projections_float);
    int projections[test_image_height][test_image_width][2];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) for (int k = 0; k < 2; k++)
        projections[i][j][k] = round(projections_float[i][j][k]);

    for (int i = 0; i < height_2; i++) for (int j = 0; j < width_2; j++)
        depth_hypothesis[0][i][j] = 0;

    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) {
        const bool is_valid_below = (projections[i][j][0] >= 0) && (projections[i][j][1] >= 0);
        const bool is_valid_above = (projections[i][j][0] < width_2) && (projections[i][j][1] < height_2);
        const bool is_valid = is_valid_below && is_valid_above;
        if (is_valid && depth_hypothesis[0][projections[i][j][1]][projections[i][j][0]] == 0)
            depth_hypothesis[0][projections[i][j][1]][projections[i][j][0]] = z_values[i][j];
    }
}


void warp_frame_depth(const float image_src[hyper_channels * 16][height_32][width_32],
                     const float depth_dst[height_32][width_32],
                     const float trans[4][4],
                     const float camera_matrix[3][3],
                     float image_dst[hyper_channels * 16][height_32][width_32]) {

    // unproject source points to camera frame
    float points_3d_dst[height_32][width_32][3];
    depth_to_3d<height_32, width_32>(depth_dst, camera_matrix, points_3d_dst);

    // apply transformation to the 3d points
    Matrix4f src_trans_dst;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) src_trans_dst(i, j) = trans[i][j];
    float points_3d_src[height_32][width_32][3];
    transform_points<height_32, width_32>(src_trans_dst, points_3d_dst, points_3d_src);
    for (int i = 0; i < height_32; i++) for (int j = 0; j < width_32; j++)
        points_3d_src[i][j][2] = max(0.0f, points_3d_src[i][j][2]);

    // project back to pixels
    float points_2d_src[height_32][width_32][2];
    project_points<height_32, width_32>(points_3d_src, camera_matrix, points_2d_src);

    // normalize points between [-1 / 1]
    float points_2d_src_norm[height_32][width_32][2];
    normalize_pixel_coordinates<height_32, width_32>(points_2d_src, points_2d_src_norm);

    grid_sample<hyper_channels * 16, height_32, width_32>(image_src, points_2d_src_norm, image_dst);
}


bool is_pose_available(const float pose[4 * 4]) {
    // is_nan = np.isnan(pose).any()
    // is_inf = np.isinf(pose).any()
    // is_neg_inf = np.isneginf(pose).any()
    // if is_nan or is_inf or is_neg_inf:
    //     return False
    // else:
    //     return True
    return true;
}