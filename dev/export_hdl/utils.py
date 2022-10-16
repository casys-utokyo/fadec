import numpy as np
import nngen as ng
import torch
import kornia

class sigmoid(ng.sigmoid):
    def __init__(self, features, par=1):
        ng.sigmoid.__init__(self, features, lut_clip=8.0, range_rate=0.5, dtype=ng.int16, par=par)


def rshift_round_and_clip(act, rshift, par, dtype):
    return ng.clip(ng.rshift_round(act, rshift, par=par), asymmetric_clip=True, par=par, dtype=dtype)


def round_and_clip(input):
    info = np.iinfo(np.int16)
    return np.clip(np.round(input).astype(np.int64), info.min, info.max).astype(np.int64)


class interpolate():
    def __init__(self, out_height, out_width, rshift, mode):
        self.out_height, self.out_width = out_height, out_width
        self.rshift = rshift
        self.mode = mode

    def __call__(self, input):
        batchsize, in_height, in_width, channels = input.shape
        out_height, out_width = self.out_height, self.out_width
        rshift = self.rshift
        mode = self.mode
        output = np.zeros((batchsize, out_height, out_width, channels), dtype=input.dtype)

        if rshift != 0:
            print("interpolate:", rshift)
        else:
            out_shape = (out_height, out_width)
            mid = torch.tensor(input.astype(np.float32).transpose(0, 3, 1, 2))
            if mode == "nearest":
                mid = torch.nn.functional.interpolate(mid, size=out_shape, mode="nearest")
            elif mode == "bilinear":
                mid = torch.nn.functional.interpolate(mid, size=out_shape, mode='bilinear', align_corners=True)
            output = mid.detach().numpy().copy().transpose(0, 2, 3, 1).astype(input.dtype)

        # if mode == "nearest":
        #     fy = in_height / out_height
        #     fx = in_width / out_width
        #     for j in range(out_height):
        #         for k in range(out_width):
        #             y = int(j * fy)
        #             x = int(k * fx)
        #             for i in range(channels):
        #                 # input_idx = (i * in_height + y) * in_width + x
        #                 # output_idx = (i * out_height + j) * out_width + k
        #                 output[0][j][k][i] = input[0][y][x][i]
        # elif mode == "bilinear":
        #     if in_height < out_height:
        #         fy = (in_height - 1) / (out_height - 1)
        #         fx = (in_width - 1) / (out_width - 1)
        #         for j in range(out_height):
        #             for k in range(out_width):
        #                 y = j * fy
        #                 x = k * fx
        #                 y_int = int(y) if y < in_height-1 else int(y) - 1
        #                 x_int = int(x) if x < in_width-1 else int(x) - 1
        #                 ys = [y_int, y_int + 1]
        #                 xs = [x_int, x_int + 1]
        #                 dys = [y - ys[0], ys[1] - y]
        #                 dxs = [x - xs[0], xs[1] - x]
        #                 for i in range(channels):
        #                     sum = 0
        #                     for yi in range(2):
        #                         for xi in range(2):
        #                             sum += dys[1-yi] * dxs[1-xi] * input[0][ys[yi]][xs[xi]][i]
        #                     output[0][j][k][i] = int(round(sum)) >> rshift
        #     else:
        #         print("in_height is larger than out_height")
        # else:
        #     print("The 'mode' option in interpolation should be 'nearest' or 'bilinear,' but it is", mode)

        return output


class lstm_state_calculator():
    def __init__(self, inputs, prepare_input_value, hshift, cshift):
        self.org_lstm_state = prepare_input_value(inputs["hidden_state"][0].transpose(0, 2, 3, 1), hshift), prepare_input_value(inputs["cell_state"][0].transpose(0, 2, 3, 1), cshift)
        self.full_K_torch = torch.tensor(inputs["full_K"])
        self.half_K_torch = torch.tensor(inputs["half_K"])
        self.camera_matrix = torch.tensor(inputs["lstm_K"])
        self.prepare_input_value = prepare_input_value
        self.hshift = hshift
        self.cshift = cshift
        self.test_image_width = 96
        self.test_image_height = 64


    def get_non_differentiable_rectangle_depth_estimation(self, reference_pose_torch, measurement_pose_torch,
                                                          previous_depth_torch, full_K_torch, half_K_torch,
                                                          original_width, original_height):
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

        depth_hypothesis = torch.zeros(size=(batch_size, 1, half_height, half_width))
        for projection_index in range(0, batch_size):
            valid_points_zs = z_values[projection_index][is_valid[projection_index]]
            valid_projections = projections[projection_index][is_valid[projection_index]]
            i_s = valid_projections[:, 1]
            j_s = valid_projections[:, 0]
            ij_combined = i_s * half_width + j_s
            _, ij_combined_unique_indices = np.unique(ij_combined.cpu().numpy(), return_index=True)
            ij_combined_unique_indices = torch.from_numpy(ij_combined_unique_indices).long()
            i_s = i_s[ij_combined_unique_indices]
            j_s = j_s[ij_combined_unique_indices]
            valid_points_zs = valid_points_zs[ij_combined_unique_indices]
            torch.index_put_(depth_hypothesis[projection_index, 0], (i_s, j_s), valid_points_zs)
        return depth_hypothesis


    def warp_frame_depth(
            self,
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


    def __call__(self, lstm_state, previous_depth, previous_pose, current_pose):
        hshift = self.hshift
        cshift = self.cshift
        prepare_input_value = self.prepare_input_value

        if lstm_state is None:
            return self.org_lstm_state
        if previous_pose is None:
            if lstm_state[0].dtype == np.int64:
                return lstm_state
            else:
                return prepare_input_value(lstm_state[0].transpose(0, 2, 3, 1), hshift), prepare_input_value(lstm_state[1].transpose(0, 2, 3, 1), cshift)

        full_K_torch = self.full_K_torch
        half_K_torch = self.half_K_torch
        camera_matrix = self.camera_matrix
        test_image_width = self.test_image_width
        test_image_height = self.test_image_height

        previous_depth = torch.tensor(previous_depth)
        previous_pose = torch.tensor(previous_pose)
        current_pose = torch.tensor(current_pose)

        if previous_depth is not None:
            depth_estimation = self.get_non_differentiable_rectangle_depth_estimation(reference_pose_torch=current_pose,
                                                                                      measurement_pose_torch=previous_pose,
                                                                                      previous_depth_torch=previous_depth,
                                                                                      full_K_torch=full_K_torch,
                                                                                      half_K_torch=half_K_torch,
                                                                                      original_height=test_image_height,
                                                                                      original_width=test_image_width)
            depth_estimation = torch.nn.functional.interpolate(input=depth_estimation,
                                                               scale_factor=(1.0 / 16.0),
                                                               mode="nearest")
        else:
            depth_estimation = torch.zeros(size=(1, 1, int(test_image_height / 32.0), int(test_image_width / 32.0)))


        if lstm_state[0].dtype == np.int64:
            h_cur = lstm_state[0].transpose(0, 3, 1, 2).astype(np.float32) / (1 << hshift)
            c_cur = lstm_state[1]
        else:
            h_cur = lstm_state[0].astype(np.float32)
            c_cur = prepare_input_value(lstm_state[1].transpose(0, 2, 3, 1), cshift)


        h_cur = torch.tensor(h_cur)
        transformation = torch.bmm(torch.inverse(previous_pose), current_pose)

        non_valid = depth_estimation <= 0.01
        h_cur = self.warp_frame_depth(image_src=h_cur,
                                      depth_dst=depth_estimation,
                                      src_trans_dst=transformation,
                                      camera_matrix=camera_matrix,
                                      normalize_points=False,
                                      sampling_mode='bilinear')
        b, c, h, w = h_cur.size()
        non_valid = torch.cat([non_valid] * c, dim=1)
        h_cur.data[non_valid] = 0.0
        return prepare_input_value(h_cur.detach().numpy().copy().transpose(0, 2, 3, 1), hshift), c_cur
