from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def celu(features, rshift_lut_in=0,
         lut_addrwidth=8, lut_clip=6.0, range_rate=0.95,
         dtype=None, name=None, par=1,
         features_dtype=None, features_scale=1, features_shamt=0):

    features_point = 0 if features_dtype is None else features_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - features_point

    rshift_lut_in_pow = np.where(rshift_lut_in > np.zeros_like(rshift_lut_in, dtype=np.int64),
                                 rshift_lut_in - 1,
                                 np.zeros_like(rshift_lut_in))
    rshift_lut_in_round = np.where(rshift_lut_in > np.zeros_like(rshift_lut_in, dtype=np.int64),
                                   np.power(np.ones_like(rshift_lut_in, dtype=np.int64) * 2,
                                            rshift_lut_in_pow),
                                   np.zeros_like(rshift_lut_in, dtype=np.int64))

    frac = np.where(rshift_lut_in!=0, np.where(features>=0, rshift_lut_in_round, rshift_lut_in_round - 1),
                    np.zeros_like(rshift_lut_in, dtype=np.int64))
    sra = np.add(features, frac)
    sra = np.right_shift(sra, rshift_lut_in)

    if dtype is None:
        raise ValueError('celu requires dtype to determine the value range.')

    out_width = dtype.width
    out_point = dtype.point
    out_signed = dtype.signed
    if out_signed:
        out_scale = round((2 ** (out_width - 1)) * range_rate)
    else:
        out_scale = round((2 ** out_width) * range_rate)

    def _celu_n(x):
        return np.around((np.exp(-x) - 1) * out_scale).astype(np.int64)

    addr_scale = lut_clip / (2 ** lut_addrwidth)
    lut = _celu_n(sra * (-addr_scale))

    p_th = 2 ** lut_addrwidth - 1
    n_th = -1 * p_th

    if out_point == 0:
        th_scale = out_scale
    elif out_point > 0:
        th_scale = out_scale >> out_point
    else:
        th_scale = out_scale << (-1 * out_point)

    n = np.where(sra < n_th, -th_scale, lut)
    out = np.where(sra >= 0, features, n)

    return out
