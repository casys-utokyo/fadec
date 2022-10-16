#include "config.h"

float KeyframeBuffer::calculate_penalty(const float t_score, const float R_score) {
    float degree = 2.0;
    float R_penalty = pow(abs(R_score - optimal_R_score), degree);
    float t_diff = t_score - optimal_t_score;
    float t_penalty;
    if (t_diff < 0.0) {
        t_penalty = 5.0 * pow(abs(t_diff), degree);
    } else {
        t_penalty = pow(abs(t_diff), degree);
    }
    return R_penalty + t_penalty;
}


int KeyframeBuffer::try_new_keyframe(const float pose[4 * 4]) {
    if (is_pose_available(pose)) {
        __tracking_lost_counter = 0;
        if (buffer_cnt == 0) {
            return 0;  // pose is available, new frame added but buffer was empty, this is the first frame, no depth map prediction will be done
        } else {
            float last_pose[4 * 4];
            const int buffer_last = (buffer_idx + buffer_cnt - 1) % buffer_size;
            for (int idx = 0; idx < 4 * 4; idx++) last_pose[idx] = buffer_poses[buffer_last][idx];

            float combined_measure, R_measure, t_measure;
            pose_distance(pose, last_pose, combined_measure, R_measure, t_measure);

            if (combined_measure >= keyframe_pose_distance)
                return 1;  // pose is available, new frame added, everything is perfect, and we will predict a depth map later
            else
                return 2;  // pose is available but not enough change has happened since the last keyframe
        }
    } else {
        __tracking_lost_counter += 1;
        if (__tracking_lost_counter > 30) {
            if (!buffer_cnt == 0) {
                buffer_idx = 0;
                buffer_cnt = 0;
                return 3;  // a pose reading has not arrived for over a second, tracking is now lost
            } else {
                return 4;  // we are still very lost
            }
        } else {
            return 5;  // pose is not available right now, but not enough time has passed to consider lost, there is still hope :)
        }
    }
}


void KeyframeBuffer::add_new_keyframe(const float pose[4 * 4], const qaint feature_half[fpn_output_channels * height_2 * width_2]) {
    const int buffer_end = (buffer_idx + buffer_cnt) % buffer_size;
    for (int idx = 0; idx < 4 * 4; idx++)
        buffer_poses[buffer_end][idx] = pose[idx];
    for (int idx = 0; idx < fpn_output_channels * height_2 * width_2; idx++)
        buffer_feature_halfs[buffer_end][idx] = feature_half[idx];
    if (buffer_cnt != buffer_size) buffer_cnt++;
}


int KeyframeBuffer::get_best_measurement_frames(const float reference_pose[4 * 4], float measurement_poses[test_n_measurement_frames * 4 * 4], qaint measurement_feature_halfs[test_keyframe_buffer_size * fpn_output_channels * height_2 * width_2]) {
    // float reference_pose[4 * 4];
    // const int buffer_last = (buffer_idx + buffer_cnt - 1) % buffer_size;
    // for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) reference_pose[i * 4 + j] = buffer_poses[buffer_last][i * 4 + j];

    if (buffer_cnt == 0) return 0;

    const int n_requested_measurement_frames = min(test_n_measurement_frames, buffer_cnt);

    pair<float, int> penalties[buffer_cnt];
    for (int bi = 0; bi < buffer_cnt; bi++) {
        const int b = (bi + buffer_idx) % buffer_size;
        float measurement_pose[4 * 4];
        for (int idx = 0; idx < 4 * 4; idx++) measurement_pose[idx] = buffer_poses[b][idx];

        float combined_measure, R_measure, t_measure;
        pose_distance(reference_pose, measurement_pose, combined_measure, R_measure, t_measure);

        float penalty = calculate_penalty(t_measure, R_measure);
        penalties[bi] = pair<float, int>(penalty, b);
    }
    sort(penalties, penalties + buffer_cnt);
    // indices = np.argpartition(penalties, n_requested_measurement_frames - 1)[:n_requested_measurement_frames]

    for (int f = 0; f < n_requested_measurement_frames; f++) {
        for (int idx = 0; idx < 4 * 4; idx++) measurement_poses[f * (4 * 4) + idx] = buffer_poses[penalties[f].second][idx];
        for (int idx = 0; idx < fpn_output_channels * height_2 * width_2; idx++)
            measurement_feature_halfs[f * (fpn_output_channels * height_2 * width_2) + idx] = buffer_feature_halfs[penalties[f].second][idx];
    }
    return n_requested_measurement_frames;
}
