import numpy as np
import nngen as ng
import torch
from utils import round_and_clip

class Fusion():
    n_depth_levels = 64

    def __init__(self, rshift, K, pose1s, pose2ss):
        self.rshift = rshift
        self.K = K[0]
        self.pose1s = pose1s[:, 0]
        self.pose2ss = pose2ss[:, :, 0]

        test_image_width = 96
        test_image_height = 64
        self.warp_grid = self.get_warp_grid_for_cost_volume_calculation(int(test_image_width / 2), int(test_image_height / 2))

        self.cython = {}


    def get_warp_grid_for_cost_volume_calculation(self, width, height):
        x = np.linspace(0, width - 1, num=int(width))
        y = np.linspace(0, height - 1, num=int(height))
        ones = np.ones(shape=(height, width))
        x_grid, y_grid = np.meshgrid(x, y)
        warp_grid = np.stack((x_grid, y_grid, ones), axis=-1)
        warp_grid = warp_grid.astype(np.float32).reshape(-1, 3).T
        return warp_grid


    def prep(self, frame_number, n_measurement_frames, image2s):
        self.n_measurement_frames = n_measurement_frames

        n_depth_levels = self.n_depth_levels
        K = self.K
        pose1 = self.pose1s[frame_number]
        pose2s = self.pose2ss[frame_number]
        warp_grid = self.warp_grid

        batchsize, height, width, channels = image2s[0].shape

        min_depth = 0.25
        max_depth = 20.0

        inverse_depth_base = 1.0 / max_depth
        inverse_depth_step = (1.0 / min_depth - 1.0 / max_depth) / (n_depth_levels - 1)

        width_normalizer = width / 2.0
        height_normalizer = height / 2.0

        self.warped_image2s = np.zeros((n_depth_levels, batchsize, height, width, channels))
        for m in range(n_measurement_frames):
            pose2 = pose2s[m]
            image2 = image2s[m]

            extrinsic2 = np.linalg.inv(pose2).dot(pose1)
            R = extrinsic2[0:3, 0:3]
            t = extrinsic2[0:3, 3]

            Kt = K.dot(t)[:, None]
            K_R_Kinv = K.dot(R).dot(np.linalg.inv(K))
            K_R_Kinv_UV = K_R_Kinv.dot(warp_grid)

            for depth_i in range(n_depth_levels):
                this_depth = 1 / (inverse_depth_base + depth_i * inverse_depth_step)

                warping = K_R_Kinv_UV + (Kt / this_depth)
                warping = warping[:2] / (warping[2] + 1e-8)
                warping = warping.T.reshape(1, height, width, 2)
                warping[:, :, :, 0] = (warping[:, :, :, 0] - width_normalizer) / width_normalizer
                warping[:, :, :, 1] = (warping[:, :, :, 1] - height_normalizer) / height_normalizer

                warped_image2 = torch.nn.functional.grid_sample(input=torch.tensor(image2.astype(np.float32).transpose(0, 3, 1, 2)),
                                                                grid=torch.tensor(warping),
                                                                mode='bilinear',
                                                                padding_mode='zeros',
                                                                align_corners=True)
                self.warped_image2s[depth_i] += warped_image2.detach().numpy().copy().transpose(0, 2, 3, 1)

        self.warped_image2s = round_and_clip(self.warped_image2s / n_measurement_frames)

        # self.cython["n_measurement_frames"] = n_measurement_frames
        # self.cython["image2s"] = image2s.astype(np.int16)
        # self.cython["K"] = K
        # self.cython["pose1"] = pose1
        # self.cython["pose2s"] = pose2s
        # self.cython["warp_grid"] = warp_grid
        # self.cython["warped_image2s"] = self.warped_image2s


    def __call__(self, image1):
        rshift = self.rshift
        fused_cost_volume = np.array([np.sum(image1 * warped_image2, axis=3) for warped_image2 in self.warped_image2s]).transpose(1, 2, 3, 0)

        # self.cython["image1"] = image1.astype(np.int16)
        # self.cython["rshift"] = rshift
        # self.cython["cost_volume"] = round_and_clip((fused_cost_volume / n_measurement_frames) / (1 << rshift), image1.dtype)
        # np.savez_compressed('pynq/cython/params.npz', **self.cython)

        return round_and_clip(fused_cost_volume / image1.shape[-1] / (1 << rshift))


# def cost_volume_fusion(act78, warped_image2s, par, K, pose1s, pose2ss, act_dtype):
def cost_volume_fusion(act78, K, pose1s, pose2ss, act_dtype):

    externs = []

    # [79] cost volume fusion
    fusion = Fusion(11, K, pose1s, pose2ss)
    act79 = ng.extern([act78], shape=(1, 32, 48, 64), dtype=act_dtype, opcode=0x79, func=fusion)
    externs.append((act79, [act78], "act79 = fusion(act78)"))
    # rshift79 = ng.constant([16], dtype=ng.int8)
    # tmp79 = ng.extern([act78], dtype=act_dtype, opcode=0x79, func=lambda x : x)
    # externs.append((tmp79, [act78], "tmp79 = act78"))
    # act79 = ng.concat([rshift_round_and_clip(ng.reduce_sum(ng.multiply(tmp79, warped_image2, dtype=ng.int32), axis=3, keep_dims=True, par=par), rshift79, par=par, dtype=act_dtype) for warped_image2 in warped_image2s], axis=3)

    return (act79,), externs, fusion

