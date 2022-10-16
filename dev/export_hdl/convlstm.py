import numpy as np
import nngen as ng
from utils import sigmoid, rshift_round_and_clip, round_and_clip

class ln():
    def __init__(self, lshift):
        self.lshift = lshift

    def __call__(self, x):
        eps = 1e-5
        e = np.mean(x.reshape(-1, x.shape[-1]), axis=0)
        v = np.var(x.reshape(-1, x.shape[-1]), axis=0)
        return round_and_clip((x - e) / np.sqrt(v + eps) * (1 << self.lshift))


def LSTMFusion(act99, act100, act101, params, par_ich, par_ochs, par,
               weight_dtype, bias_dtype, scale_dtype, act_dtype, mid_dtype):

    externs = []

    # [102] cat
    tmp102 = ng.extern([act100, act99], opcode=0x102, func=lambda x, y : x)
    externs.append((tmp102, [act100], "tmp102 = act100"))
    act102 = ng.concat([act99, tmp102], axis=3)
    # act102 = ng.concat([act99, act100], axis=3)


    # [103] conv
    weight103 = ng.variable(dtype=weight_dtype, shape=(2048, 3, 3, 1024), name="lstm_cell.conv.weight")
    weight103.set_value(params["lstm_cell.conv.weight"])

    rshift103 = ng.constant([11], dtype=ng.int8)
    act103 = ng.conv2d(act102, weight103, strides=(1, 1, 1, 1), rshift_out=rshift103, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [104] sig_ln_celu
    slice104s = [ng.slice_(act103, (0, 0, 0, i * 512), (1, 2, 3, (i+1) * 512), (1, 1, 1, 1)) for i in range(4)]

    rshift104 = ng.constant([4], dtype=ng.int8)
    ii104, ff104, oo104 = [sigmoid(ng.rshift_round(slice104s[i], rshift104, par=par), par=par) for i in range(3)]

    ln104 = ng.extern([slice104s[3]], opcode=0x104, func=ln(12))
    externs.append((ln104, [slice104s[3]], "ln104 = ln(12)(slice104s[3])"))
    gg104 = ng.celu(ln104, rshift_lut_in=7, lut_clip=8.0, range_rate=0.125, dtype=act_dtype, par=par)


    # [105] cell_state
    in_rshift105 = ng.constant([2], dtype=ng.int8)
    out_rshift105 = ng.constant([12], dtype=ng.int8)
    sum105 = rshift_round_and_clip(ng.add(ng.multiply(ng.rshift_round(ff104, in_rshift105, par=par), act101, par=par, dtype=mid_dtype), ng.multiply(ng.rshift_round(ii104, in_rshift105, par=par), gg104, par=par, dtype=mid_dtype), par=par), out_rshift105, par=par, dtype=act_dtype)
    act105 = ng.extern([sum105], opcode=0x105, func=ln(12))
    externs.append((act105, [sum105], "act105 = ln(12)(sum105)"))


    # [106] hidden_state
    celu106 = ng.celu(act105, rshift_lut_in=7, lut_clip=8.0, range_rate=0.125, dtype=act_dtype, par=par)
    rshift106 = ng.constant([13], dtype=ng.int8)
    act106 = rshift_round_and_clip(ng.multiply(celu106, oo104, par=par, dtype=mid_dtype), rshift106, par=par, dtype=act_dtype)


    return (act106, act105), externs
