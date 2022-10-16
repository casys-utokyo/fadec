# distutils: language=c++
# distutils: extra_compile_args = ["-O3", "-fopenmp"]
# distutils: extra_link_args=["-fopenmp"]
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
# cython: cdivision=True

import numpy as np
from libc.math cimport round
from cython.parallel import prange


cpdef float[:, :, :, :, :] prep_cython(char n_measurement_frames, short[:, :, :, :, :] image2s,
                                       float[:, :] K, float[:, :] inv_K,
                                       float[:, :] pose1, float[:, :, :] inv_pose2s, float[:, :] warp_grid):

    cdef char batchsize = 1
    cdef char height = 32
    cdef char width = 48
    cdef char channels = 32
    cdef char n_depth_levels = 64

    cdef float min_depth = 0.25
    cdef float max_depth = 20.0

    cdef float inverse_depth_base = 1.0 / max_depth
    cdef float inverse_depth_step = (1.0 / min_depth - 1.0 / max_depth) / (n_depth_levels - 1)

    cdef float width_normalizer = width / 2.0
    cdef float height_normalizer = height / 2.0

    cdef float[:, :, :, :, :] warped_image2s = np.zeros((n_depth_levels, batchsize, height, width, channels), dtype=np.float32)

    cdef float[3][3] R
    cdef float[3] t
    cdef float[3] Kt
    cdef float[3][3] K_R
    cdef float[3][3] K_R_Kinv
    cdef float[1536][3] K_R_Kinv_UV

    cdef float this_depth
    cdef float[3] Ktd
    cdef float w2

    cdef short warp0
    cdef short warp1

    cdef short m
    cdef short i
    cdef short j
    cdef short k
    cdef short depth_i

    cdef short height1 = height-1
    cdef short width1 = width-1

    cdef short x_int
    cdef short y_int

    cdef short xs0
    cdef short xs1
    cdef short ys0
    cdef short ys1
    cdef short dxs0
    cdef short dxs1
    cdef short dys0
    cdef short dys1

    cdef char xi
    cdef char yi
    cdef char rshift = ((n_measurement_frames-1) + 8 * 2)
    cdef int s

    for m in range(n_measurement_frames):
        for i in range(3):
            for j in range(3):
                R[i][j] = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    R[i][j] += inv_pose2s[m][i][k] * pose1[k][j]


        for i in range(3):
            t[i] = 0
        for i in range(3):
            for j in range(3):
                t[i] += inv_pose2s[m][i][j] * pose1[j][3]
            t[i] += inv_pose2s[m][i][3]

        for i in range(3):
            Kt[i] = 0
        for i in range(2):
            for j in range(3):
                Kt[i] += K[i][j] * t[j]
        Kt[2] = t[2]

        for i in range(3):
            for j in range(3):
                K_R[i][j] = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    K_R[i][j] += K[i][k] * R[k][j]

        for i in range(3):
            for j in range(3):
                K_R_Kinv[i][j] = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    K_R_Kinv[i][j] += K_R[i][k] * inv_K[k][j]

        for j in range(1536):
            for i in range(3):
                K_R_Kinv_UV[j][i] = 0
        for j in range(1536):
            for i in range(3):
                for k in range(3):
                    K_R_Kinv_UV[j][i] += K_R_Kinv[i][k] * warp_grid[k][j]


        for depth_i in range(n_depth_levels):
            this_depth = inverse_depth_base + depth_i * inverse_depth_step

            for i in range(3):
                Ktd[i] = Kt[i] * this_depth

            for i in prange(height, nogil=True, schedule='static', chunksize=1):
            # for i in range(height):
                for j in range(width):

                    k = i * width + j
                    w2 = K_R_Kinv_UV[k][2] + Ktd[2] + 1e-8
                    warp0 = <short>round(((K_R_Kinv_UV[k][0] + Ktd[0]) / (w2 * width_normalizer)) * (width1 / 2.0 * (1 << 8)))
                    warp1 = <short>round(((K_R_Kinv_UV[k][1] + Ktd[1]) / (w2 * height_normalizer)) * (height1 / 2.0 * (1 << 8)))

                    x_int = warp0 >> 8
                    y_int = warp1 >> 8
                    xs0 = x_int
                    xs1 = x_int + 1
                    ys0 = y_int
                    ys1 = y_int + 1
                    dxs0 = warp0 & ((1 << 8) - 1)
                    dys0 = warp1 & ((1 << 8) - 1)
                    dxs1 = (1 << 8) - dxs0
                    dys1 = (1 << 8) - dys0
                    for k in range(channels):
                        s = 1 << (rshift-1)
                        if not(ys0 < 0 or height1 < ys0 or xs0 < 0 or width1 < xs0):
                            s += dys1 * dxs1 * image2s[m][0][ys0][xs0][k]
                        if not(ys0 < 0 or height1 < ys0 or xs1 < 0 or width1 < xs1):
                            s += dys1 * dxs0 * image2s[m][0][ys0][xs1][k]
                        if not(ys1 < 0 or height1 < ys1 or xs0 < 0 or width1 < xs0):
                            s += dys0 * dxs1 * image2s[m][0][ys1][xs0][k]
                        if not(ys1 < 0 or height1 < ys1 or xs1 < 0 or width1 < xs1):
                            s += dys0 * dxs0 * image2s[m][0][ys1][xs1][k]
                        warped_image2s[depth_i][0][i][j][k] += <short>(s >> rshift)

    return warped_image2s


cpdef short[:, :, :, :] fusion_quantize_cython(short[:, :, :, :] image1, short[:, :, :, :, :] warped_image2s):

    cdef char batchsize = 1
    cdef char height = 32
    cdef char width = 48
    cdef char channels = 32
    cdef char n_depth_levels = 64
    cdef char rshift = 16

    cdef short short_max = 32767

    cdef short[:, :, :, :] fused_cost_volume = np.zeros((batchsize, height, width, n_depth_levels), dtype=np.int16)
    cdef int s
    cdef char depth_i
    cdef char b
    cdef char h
    cdef char w
    cdef char c

    for depth_i in prange(n_depth_levels, nogil=True, schedule='static', chunksize=1):
    # for depth_i in range(n_depth_levels):
        for b in range(batchsize):
            for h in range(height):
                for w in range(width):
                    s = (1 << (rshift-1))
                    for c in range(channels):
                        s += image1[b][h][w][c] * warped_image2s[depth_i][b][h][w][c]
                    s = s >> rshift
                    fused_cost_volume[b][h][w][depth_i] = <short>s

    return fused_cost_volume
