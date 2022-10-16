import numpy as np
from path import Path
import os
import struct

INTMAX = [None, 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071, 262143, 524287, 1048575, 2097151, 4194303, 8388607, 16777215, 33554431, 67108863, 134217727, 268435455, 536870911, 1073741823, 2147483647]
INTMIN = [None, -1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024, -2048, -4096, -8192, -16384, -32768, -65536, -131072, -262144, -524288, -1048576, -2097152, -4194304, -8388608, -16777216, -33554432, -67108864, -134217728, -268435456, -536870912, -1073741824, -2147483648]

act_cnt = 0
ln_aves = []
ln_inv_stds = []

def quantize(act, bit, alpha=0.95):
    global act_cnt

    param = act[1].copy()
    if act[0] == "add":
        param.append(param[0] + param[1])

    param = [np.abs(p.reshape(-1)) for p in param]

    if act[0] in ["add", "conv", "relu"]:
        param = [np.sort(p) for p in param]
        idx = [int(round(len(p) * alpha)) for p in param]
        scale = [float(INTMAX[bit-1] / p[i]) for p, i in zip(param, idx)]
        shift = [int(np.floor(np.log2(s))) for s in scale]
        print(act[0], [p[i] for p, i in zip(param, idx)], shift)
        return shift
    elif act[0] in ["interpolate", "cost_volume", "cat", "layer_norm", "cell_hidden"]:
        param = [np.sort(p) for p in param]
        idx = [int(round(len(p) * alpha)) for p in param]
        scale = [float(INTMAX[bit-1] / p[i]) for p, i in zip(param, idx)]
        shift = [int(np.floor(np.log2(s))) for s in scale]
        print(act[0], [p[i] for p, i in zip(param, idx)], shift)
        if act[0] == "layer_norm":
            ln = act[1][-2].copy().transpose(2, 0, 1, 3, 4)
            ln = ln.reshape(ln.shape[0], ln.shape[1], -1)
            ln_ave = np.mean(ln, axis=2)
            ln_var = np.var(ln, axis=2)
            ln_aves.append(np.round(np.mean(ln_ave, axis=1) * (1 << shift[0])).astype('int32'))
            ln_inv_stds.append(np.round(1 / np.mean(np.sqrt(ln_var), axis=1) * (1 << shift[-1])).astype('int32'))
        return shift
    elif act[0] in ["sigmoid", "celu"]:
        act_cnt += 1
        print("%7s: %.5f, %.5f" % (act[0], np.max(param[0]), np.max(param[1])))
        return None
    else:
        print(act[0])
        return None


if __name__ == '__main__':
    print("reading activation file...")
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    param_dir = base_dir / "../params"
    npz_acts = np.load(param_dir / "tmp/acts.npz", allow_pickle=True)
    acts = npz_acts["acts"]
    bit = 16

    print("quantizing...")
    param_cpp_dir = param_dir / "params_cpp_with_ptq"
    fs = [[open(param_cpp_dir / "cin_shifts", "wb"), open(param_cpp_dir / "cout_shifts", "wb")],
          [open(param_cpp_dir / "ain1_shifts", "wb"), open(param_cpp_dir / "ain2_shifts", "wb"), open(param_cpp_dir / "aout_shifts", "wb")],
          [open(param_cpp_dir / "oin_shifts", "wb"), open(param_cpp_dir / "oout_shifts", "wb")]]
    cnt = [0, 0, 0]
    shifts = []
    for act in acts:
        shift = quantize(act, bit)
        if act[0] in ["interpolate", "cost_volume", "cat", "layer_norm", "cell_hidden"]:
            shifts.append((act[0], shift))
        elif shift is not None:
            assert len(shift) == 2 or len(shift) == 3
            idx = 0 if act[0] == "conv" else 1 if act[0] == "add" else 2
            for s, f in zip(shift, fs[idx]):
                f.write(struct.pack('i', s))
            cnt[idx] += 1

    for i in range(len(fs)):
        for j in range(len(fs[i])):
            fs[i][j].close()
    print(cnt)
    print(shifts)
