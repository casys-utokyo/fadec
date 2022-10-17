from __future__ import division

import numpy as np
import torch

# GEOMETRIC UTILS
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
