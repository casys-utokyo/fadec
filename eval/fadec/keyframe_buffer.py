from collections import deque

import numpy as np

def pose_distance(reference_pose, measurement_pose):
    """
    :param reference_pose: 4x4 numpy array, reference frame camera-to-world pose (not extrinsic matrix!)
    :param measurement_pose: 4x4 numpy array, measurement frame camera-to-world pose (not extrinsic matrix!)
    :return combined_measure: float, combined pose distance measure
    :return R_measure: float, rotation distance measure
    :return t_measure: float, translation distance measure
    """
    rel_pose = np.dot(np.linalg.inv(reference_pose), measurement_pose)
    R = rel_pose[:3, :3]
    t = rel_pose[:3, 3]
    R_measure = np.sqrt(2 * (1 - min(3.0, np.matrix.trace(R)) / 3))
    t_measure = np.linalg.norm(t)
    combined_measure = np.sqrt(t_measure ** 2 + R_measure ** 2)
    return combined_measure, R_measure, t_measure


def is_pose_available(pose):
    is_nan = np.isnan(pose).any()
    is_inf = np.isinf(pose).any()
    is_neg_inf = np.isneginf(pose).any()
    if is_nan or is_inf or is_neg_inf:
        return False
    else:
        return True


class KeyframeBuffer:
    def __init__(self, buffer_size, keyframe_pose_distance, optimal_t_score, optimal_R_score, store_return_indices):
        self.buffer = deque([], maxlen=buffer_size)
        self.keyframe_pose_distance = keyframe_pose_distance
        self.last_pose = None
        self.optimal_t_score = optimal_t_score
        self.optimal_R_score = optimal_R_score
        self.__tracking_lost_counter = 0
        self.__store_return_indices = store_return_indices  # mostly required for simulation of the frame selection

    def calculate_penalty(self, t_score, R_score):
        degree = 2.0
        R_penalty = np.abs(R_score - self.optimal_R_score) ** degree
        t_diff = t_score - self.optimal_t_score
        if t_diff < 0.0:
            t_penalty = 5.0 * (np.abs(t_diff) ** degree)
        else:
            t_penalty = np.abs(t_diff) ** degree
        return R_penalty + t_penalty

    def try_new_keyframe(self, pose):
        if is_pose_available(pose):
            self.__tracking_lost_counter = 0
            if len(self.buffer) == 0:
                self.last_pose = pose
                return 0  # pose is available, new frame added but buffer was empty, this is the first frame, no depth map prediction will be done
            else:
                # last_pose = self.buffer[-1][0]

                combined_measure, R_measure, t_measure = pose_distance(pose, self.last_pose)

                if combined_measure >= self.keyframe_pose_distance:
                    self.last_pose = pose
                    return 1  # pose is available, new frame added, everything is perfect, and we will predict a depth map later
                else:
                    return 2  # pose is available but not enough change has happened since the last keyframe
        else:
            self.__tracking_lost_counter += 1

            if self.__tracking_lost_counter > 30:
                if len(self.buffer) > 0:
                    self.buffer.clear()
                    return 3  # a pose reading has not arrived for over a second, tracking is now lost
                else:
                    return 4  # we are still very lost
            else:
                return 5  # pose is not available right now, but not enough time has passed to consider lost, there is still hope :)

    def add_new_keyframe(self, pose, image, index=None):
        if self.__store_return_indices and index is None:
            raise ValueError("Storing and returning the frame indices is requested in the constructor, but index=None is passed to the function")

        if self.__store_return_indices:
            self.buffer.append((pose, image, index))
        else:
            self.buffer.append((pose, image))

    def get_best_measurement_frames(self, reference_pose, n_requested_measurement_frames):
        buffer_array = list(self.buffer)

        # if self.__store_return_indices:
        #     reference_pose, reference_image, reference_index = buffer_array[-1]
        # else:
        #     reference_pose, reference_image = buffer_array[-1]

        n_requested_measurement_frames = min(n_requested_measurement_frames, len(buffer_array))

        penalties = []
        for i in range(len(buffer_array)):
            measurement_pose = buffer_array[i][0]
            combined_measure, R_measure, t_measure = pose_distance(reference_pose, measurement_pose)
            penalty = self.calculate_penalty(t_measure, R_measure)
            penalties.append(penalty)
        indices = np.argpartition(penalties, n_requested_measurement_frames - 1)[:n_requested_measurement_frames]

        measurement_frames = []
        for index in indices:
            measurement_frames.append(buffer_array[index])
        return measurement_frames
