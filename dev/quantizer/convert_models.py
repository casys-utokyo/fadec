import torch
from path import Path

import os
import struct
from config import Config

def convert():
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    fusionnet_test_weights = base_dir / Config.fusionnet_test_weights
    checkpoints = sorted(Path(fusionnet_test_weights).files())

    param_dir = base_dir / "../params"
    bin_dir = param_dir / "org_weights_bin"

    for checkpoint in checkpoints:
        save_dir = bin_dir / checkpoint.split("/")[-1]
        os.makedirs(save_dir, exist_ok=True)

        weights = torch.load(checkpoint)
        for key in weights:
            val = weights[key].to('cpu').detach().numpy().copy().reshape(-1)
            if val.dtype == 'float32':
                print("Saving  : %s" % key)
                d = bytearray()
                for v in val:
                    d += struct.pack('f', v)
                with open(save_dir / Path(key), 'wb') as f:
                    f.write(d)
            else:
                print("Skipping: %s" % key)

    name_dir = param_dir / "param_names"
    checkpoints = sorted(name_dir.files())

    param_cpp_dir = param_dir / "params_cpp"
    os.makedirs(param_cpp_dir, exist_ok=True)

    fpp = open(param_cpp_dir / "params", "wb")
    fvv = open(param_cpp_dir / "values", "wb")

    cnts = []
    for checkpoint in checkpoints:
        with open(name_dir / checkpoint.name, 'r') as f:
            params = f.read().split()

        cnt = 0
        for param in params:
            with open(bin_dir / checkpoint.name / param, "rb") as ft:
                data = ft.read()
            fpp.write(data)
            fvv.write(struct.pack('i', len(data) // 4))
            cnt += len(data) // 4
        cnts.append(cnt)

    print(cnts)
    fpp.close()
    fvv.close()

if __name__ == '__main__':
    convert()
