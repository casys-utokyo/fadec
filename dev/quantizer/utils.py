from __future__ import division

import kornia2 as kornia
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


def get_warp_grid_for_cost_volume_calculation(width, height, device):
    x = np.linspace(0, width - 1, num=int(width))
    y = np.linspace(0, height - 1, num=int(height))
    ones = np.ones(shape=(height, width))
    x_grid, y_grid = np.meshgrid(x, y)
    warp_grid = np.stack((x_grid, y_grid, ones), axis=-1)
    warp_grid = torch.from_numpy(warp_grid).float()
    warp_grid = warp_grid.view(-1, 3).t().to(device)
    return warp_grid


def calculate_cost_volume_by_warping(image1, image2, pose1, pose2, K, warp_grid, min_depth, max_depth, n_depth_levels, device, dot_product):
    batch_size, channels, height, width = image1.size()
    warp_grid = torch.cat(batch_size * [warp_grid.unsqueeze(dim=0)])

    cost_volume = torch.empty(size=(batch_size, n_depth_levels, height, width), dtype=torch.float32).to(device)

    extrinsic2 = torch.inverse(pose2).bmm(pose1)
    R = extrinsic2[:, 0:3, 0:3]
    t = extrinsic2[:, 0:3, 3].unsqueeze(-1)

    Kt = K.bmm(t)
    K_R_Kinv = K.bmm(R).bmm(torch.inverse(K))
    K_R_Kinv_UV = K_R_Kinv.bmm(warp_grid)

    inverse_depth_base = 1.0 / max_depth
    inverse_depth_step = (1.0 / min_depth - 1.0 / max_depth) / (n_depth_levels - 1)

    width_normalizer = width / 2.0
    height_normalizer = height / 2.0

    for depth_i in range(n_depth_levels):
        this_depth = 1 / (inverse_depth_base + depth_i * inverse_depth_step)

        warping = K_R_Kinv_UV + (Kt / this_depth)
        warping = warping.transpose(dim0=1, dim1=2)
        warping = warping[:, :, 0:2] / (warping[:, :, 2].unsqueeze(-1) + 1e-8)
        warping = warping.view(batch_size, height, width, 2)
        warping[:, :, :, 0] = (warping[:, :, :, 0] - width_normalizer) / width_normalizer
        warping[:, :, :, 1] = (warping[:, :, :, 1] - height_normalizer) / height_normalizer

        warped_image2 = torch.nn.functional.grid_sample(input=image2,
                                                        grid=warping,
                                                        mode='bilinear',
                                                        padding_mode='zeros',
                                                        align_corners=True)

        if dot_product:
            cost_volume[:, depth_i, :, :] = torch.sum(image1 * warped_image2, dim=1) / channels
        else:
            cost_volume[:, depth_i, :, :] = torch.sum(torch.abs(image1 - warped_image2), dim=1)

    return cost_volume


def cost_volume_fusion(image1, image2s, pose1, pose2s, K, warp_grid, min_depth, max_depth, n_depth_levels, device, dot_product):
    batch_size, channels, height, width = image1.size()
    fused_cost_volume = torch.zeros(size=(batch_size, n_depth_levels, height, width), dtype=torch.float32).to(device)

    for pose2, image2 in zip(pose2s, image2s):
        cost_volume = calculate_cost_volume_by_warping(image1=image1,
                                                       image2=image2,
                                                       pose1=pose1,
                                                       pose2=pose2,
                                                       K=K,
                                                       warp_grid=warp_grid,
                                                       min_depth=min_depth,
                                                       max_depth=max_depth,
                                                       n_depth_levels=n_depth_levels,
                                                       device=device,
                                                       dot_product=dot_product)
        fused_cost_volume += cost_volume
    fused_cost_volume /= len(pose2s)
    return fused_cost_volume


def get_non_differentiable_rectangle_depth_estimation(reference_pose_torch,
                                                      measurement_pose_torch,
                                                      previous_depth_torch,
                                                      full_K_torch,
                                                      half_K_torch,
                                                      original_width,
                                                      original_height):
    batch_size, _, _ = reference_pose_torch.shape
    half_width = int(original_width / 2)
    half_height = int(original_height / 2)

    trans = torch.bmm(torch.inverse(reference_pose_torch), measurement_pose_torch)
    points_3d_src = kornia.depth_to_3d(previous_depth_torch, full_K_torch, normalize_points=False)
    points_3d_src = points_3d_src.permute(0, 2, 3, 1)
    points_3d_dst = kornia.transform_points(trans[:, None], points_3d_src)

    points_3d_dst = points_3d_dst.view(batch_size, -1, 3)

    z_values = points_3d_dst[:, :, -1]
    z_values = torch.relu(z_values)
    sorting_indices = torch.argsort(z_values, descending=True)
    z_values = torch.gather(z_values, dim=1, index=sorting_indices)

    sorting_indices_for_points = torch.stack([sorting_indices] * 3, dim=-1)
    points_3d_dst = torch.gather(points_3d_dst, dim=1, index=sorting_indices_for_points)

    projections = torch.round(kornia.project_points(points_3d_dst, half_K_torch.unsqueeze(1))).long()
    is_valid_below = (projections[:, :, 0] >= 0) & (projections[:, :, 1] >= 0)
    is_valid_above = (projections[:, :, 0] < half_width) & (projections[:, :, 1] < half_height)
    is_valid = is_valid_below & is_valid_above

    depth_hypothesis = torch.zeros(size=(batch_size, 1, half_height, half_width)).cuda()
    for projection_index in range(0, batch_size):
        valid_points_zs = z_values[projection_index][is_valid[projection_index]]
        valid_projections = projections[projection_index][is_valid[projection_index]]
        i_s = valid_projections[:, 1]
        j_s = valid_projections[:, 0]
        ij_combined = i_s * half_width + j_s
        _, ij_combined_unique_indices = np.unique(ij_combined.cpu().numpy(), return_index=True)
        ij_combined_unique_indices = torch.from_numpy(ij_combined_unique_indices).long().cuda()
        i_s = i_s[ij_combined_unique_indices]
        j_s = j_s[ij_combined_unique_indices]
        valid_points_zs = valid_points_zs[ij_combined_unique_indices]
        torch.index_put_(depth_hypothesis[projection_index, 0], (i_s, j_s), valid_points_zs)
    return depth_hypothesis


def warp_frame_depth(
        image_src: torch.Tensor,
        depth_dst: torch.Tensor,
        src_trans_dst: torch.Tensor,
        camera_matrix: torch.Tensor,
        normalize_points: bool = False,
        sampling_mode='bilinear') -> torch.Tensor:
    # TAKEN FROM KORNIA LIBRARY
    if not isinstance(image_src, torch.Tensor):
        raise TypeError(f"Input image_src type is not a torch.Tensor. Got {type(image_src)}.")

    if not len(image_src.shape) == 4:
        raise ValueError(f"Input image_src musth have a shape (B, D, H, W). Got: {image_src.shape}")

    if not isinstance(depth_dst, torch.Tensor):
        raise TypeError(f"Input depht_dst type is not a torch.Tensor. Got {type(depth_dst)}.")

    if not len(depth_dst.shape) == 4 and depth_dst.shape[-3] == 1:
        raise ValueError(f"Input depth_dst musth have a shape (B, 1, H, W). Got: {depth_dst.shape}")

    if not isinstance(src_trans_dst, torch.Tensor):
        raise TypeError(f"Input src_trans_dst type is not a torch.Tensor. "
                        f"Got {type(src_trans_dst)}.")

    if not len(src_trans_dst.shape) == 3 and src_trans_dst.shape[-2:] == (3, 3):
        raise ValueError(f"Input src_trans_dst must have a shape (B, 3, 3). "
                         f"Got: {src_trans_dst.shape}.")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. "
                        f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")
    # unproject source points to camera frame
    points_3d_dst: torch.Tensor = kornia.depth_to_3d(depth_dst, camera_matrix, normalize_points)  # Bx3xHxW

    # transform points from source to destination
    points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3

    # apply transformation to the 3d points
    points_3d_src = kornia.transform_points(src_trans_dst[:, None], points_3d_dst)  # BxHxWx3
    points_3d_src[:, :, :, 2] = torch.relu(points_3d_src[:, :, :, 2])

    # project back to pixels
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1xHxW
    points_2d_src: torch.Tensor = kornia.project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

    # normalize points between [-1 / 1]
    height, width = depth_dst.shape[-2:]
    points_2d_src_norm: torch.Tensor = kornia.normalize_pixel_coordinates(points_2d_src, height, width)  # BxHxWx2

    return torch.nn.functional.grid_sample(image_src, points_2d_src_norm, align_corners=True, mode=sampling_mode)


def is_pose_available(pose):
    is_nan = np.isnan(pose).any()
    is_inf = np.isinf(pose).any()
    is_neg_inf = np.isneginf(pose).any()
    if is_nan or is_inf or is_neg_inf:
        return False
    else:
        return True
