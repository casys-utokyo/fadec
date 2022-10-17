import torch
import numpy as np
import os
from path import Path
import struct
from config import Config

INTMAX = [None, 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071, 262143, 524287, 1048575, 2097151, 4194303, 8388607, 16777215, 33554431, 67108863, 134217727, 268435455, 536870911, 1073741823, 2147483647]
INTMIN = [None, -1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024, -2048, -4096, -8192, -16384, -32768, -65536, -131072, -262144, -524288, -1048576, -2097152, -4194304, -8388608, -16777216, -33554432, -67108864, -134217728, -268435456, -536870912, -1073741824, -2147483648]

def quantize(param, bit):
    param = param.reshape(-1)
    m = max(-np.min(param), np.max(param))
    if m == 0:
        return np.inf, 32, param
    scale = float(INTMAX[bit] / m)
    shift = int(np.floor(np.log2(scale)))
    scaled_param = param * (2 ** shift)
    return scale, shift, np.round(scaled_param)


def quantize_save(params_out, bit, fps):
    cnt = 0
    for param in params_out:
        byte = 1 if bit <= 8 else 2 if bit <= 16 else 4
        _, shift, scaled_param = quantize(param, bit)
        scaled_param = scaled_param.astype('int%d' % (byte * 8))

        d = bytearray()
        fps[0].write(struct.pack('i', len(scaled_param)))
        fmt = [None, 'b', 'h', None, 'i']
        for p in scaled_param:
            d += struct.pack(fmt[byte], p)
        fps[1].write(d)
        fps[2].write(struct.pack('i', shift))
        cnt += len(scaled_param)
    return cnt


def main():
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    fusionnet_test_weights = base_dir / Config.fusionnet_test_weights
    checkpoints = sorted(Path(fusionnet_test_weights).files())

    param_dir = base_dir / "../params"
    param_cpp_dir = param_dir / "params_cpp_with_ptq"
    os.makedirs(param_cpp_dir, exist_ok=True)

    fw = [open(param_cpp_dir / "n_weights", "wb"),
          open(param_cpp_dir / "weights_quantized", "wb"),
          open(param_cpp_dir / "weight_shifts", "wb")]
    wbit = 8

    fb = [open(param_cpp_dir / "n_biases", "wb"),
          open(param_cpp_dir / "biases_quantized", "wb"),
          open(param_cpp_dir / "bias_shifts", "wb")]
    bbit = 28

    fs = [open(param_cpp_dir / "n_scales", "wb"),
          open(param_cpp_dir / "scales_quantized", "wb"),
          open(param_cpp_dir / "scale_shifts", "wb")]
    sbit = 8

    weights_out = []
    biases_out = []
    scales_out = []
    params_out = {}
    for checkpoint in checkpoints:
        print("Processing %s..." % checkpoint.name)
        with open(param_dir / "param_names" / checkpoint.name, 'r') as f:
            files = f.read().split()

        weights = torch.load(checkpoint)
        params = [weights[key].cpu().detach().numpy().copy() for key in files]
        idx = 0
        while idx < len(files):
            if ".running_mean" in files[idx]:
                assert ".weight" in files[idx-1]

                running_mean = params[idx]
                running_var = params[idx+1]
                weight = params[idx+2]
                bias = params[idx+3]

                wrv = weight / np.sqrt(running_var + 1e-5)
                scales_out.append(wrv)
                params_out[files[idx-1][:-7] + ".scale"] = wrv
                biases_out.append(bias / wrv - running_mean)
                params_out[files[idx-1][:-7] + ".bias"] = bias / wrv - running_mean
                idx += 4
            elif ".weight" in files[idx]:
                weights_out.append(params[idx])
                params_out[files[idx]] = params[idx]
                idx += 1
            elif ".bias" in files[idx]:
                biases_out.append(params[idx])
                params_out[files[idx]] = params[idx]
                idx += 1

        if checkpoint.name == "3_lstm_fusion":
            biases_out.append(np.zeros(params[idx-1].shape[0]))

    wcnt = quantize_save(weights_out, wbit, fw)
    bcnt = quantize_save(biases_out, bbit, fb)
    scnt = quantize_save(scales_out, sbit, fs)

    print(len(weights_out), wcnt)
    print(len(biases_out), bcnt)
    print(len(scales_out), scnt)

    for key in params_out.keys():
        bit = wbit if ".weight" in key else bbit if ".bias" in key else sbit
        byte = 1 if bit <= 8 else 2 if bit <= 16 else 4
        shape = params_out[key].shape
        params_out[key] = quantize(params_out[key], bit)[-1].astype('int%d' % (byte * 8)).reshape(shape)
        if ".weight" in key:
            params_out[key] = params_out[key].transpose(0, 2, 3, 1)

    quantized_dir = param_dir / "quantized_params"
    os.makedirs(quantized_dir, exist_ok=True)
    np.savez_compressed(quantized_dir / "params", **params_out)


if __name__ == '__main__':
    main()
