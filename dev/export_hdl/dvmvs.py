from __future__ import absolute_import
from __future__ import print_function

import time
import os
from unittest import skip
from xml.sax.handler import feature_external_ges

import numpy as np
import nngen as ng

from feature_extractor import feature_extractor
from feature_shrinker import feature_shrinker
from cost_volume_fusion import cost_volume_fusion
from cost_volume_encoder import cost_volume_encoder
from convlstm import LSTMFusion
from cost_volume_decoder import cost_volume_decoder

from verify import Verifier


def prepare_placeholders(batchsize, act_dtype):
    print("preparing placeholders...")

    reference_image = ng.placeholder(dtype=act_dtype, shape=(batchsize, 64, 96, 3), name='reference_image')
    hidden_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='hidden_state')
    cell_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='cell_state')

    return reference_image, hidden_state, cell_state


def prepare_nets(reference_image, hidden_state, cell_state, pars, dtypes):

    externs = []

    print("preparing feature extractor...")
    layers = feature_extractor(reference_image, params, **pars, **dtypes)

    print("preparing feature shrinker...")
    reference_features, extern = feature_shrinker(*layers, params, **pars, **dtypes)
    externs.extend(extern)

    print("preparing cost volume fusion...")
    cost_volume, extern, fusion = cost_volume_fusion(reference_features[0], inputs["half_K"],
                                                     inputs["current_pose"], inputs["measurement_poses"], dtypes["act_dtype"])
    externs.extend(extern)

    print("preparing cost volume encoder...")
    skips = cost_volume_encoder(*reference_features, *cost_volume, params, **pars, **dtypes)

    print("preparing LSTM fusion...")
    lstm_states, extern = LSTMFusion(skips[-1], hidden_state, cell_state, params, **pars, **dtypes)
    externs.extend(extern)

    print("preparing cost volume decoder...")
    depth_full, extern = cost_volume_decoder(reference_image, *skips[:-1], lstm_states[0], params, **pars, **dtypes)
    externs.extend(extern)

    return (layers, reference_features, cost_volume, skips, lstm_states, depth_full), externs, fusion


if __name__ == '__main__':
    batchsize = 1
    max_n_measurement_frames = 2
    project_name = "dvmvs"

    par_ich = 2
    # (kernel_size, stride): 38, 34, 5, 15, 4 (total: 96)
    par_ochs = {(1, 1): 4, (3, 1): 4, (3, 2): 4, (5, 1): 2, (5, 2): 2}
    par = 4
    pars = {"par_ich": par_ich, "par_ochs": par_ochs, "par": par}

    weight_dtype = ng.int8
    bias_dtype = ng.int32
    scale_dtype = ng.int8
    act_dtype = ng.int16
    mid_dtype = ng.int32
    dtypes = {"weight_dtype": weight_dtype, "bias_dtype": bias_dtype,
              "scale_dtype": scale_dtype, "act_dtype": act_dtype, "mid_dtype": mid_dtype}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    params = np.load(os.path.join(base_dir, "../params/quantized_params/params.npz"))
    inputs = np.load(os.path.join(base_dir, "../params/quantized_params/inputs.npz"))
    outputs = np.load(os.path.join(base_dir, "../params/quantized_params/outputs.npz"))

    start_time = time.process_time()
    input_layers = prepare_placeholders(batchsize, act_dtype)
    nets, externs, fusion = prepare_nets(*input_layers, pars, dtypes)
    print("\t%f [s]" % (time.process_time() - start_time))


    skip_verify = False
    if not(skip_verify):
        print("verifying...")
        verifier = Verifier(inputs, outputs, max_n_measurement_frames, fusion, act_dtype)
        input_layer_values, output_layer_values, output_layers = verifier.verify_one(*nets, verbose=True)

    output_layers = [nets[1][0], nets[-2][1], nets[-2][0], nets[-1][0]]

    skip_to_ipxact = False
    axi_datawidth = 128
    if skip_to_ipxact:
        print("skiping to_ipxact...")
    else:
        # Convert the NNgen dataflow to a hardware description (Verilog HDL and IP-XACT)
        print("converting NNgen dataflow to hardware description...")
        # to IP-XACT (the method returns Veriloggen object, as well as to_veriloggen)
        start_time = time.process_time()
        targ = ng.to_ipxact(output_layers, project_name, silent=False,
                            config={'maxi_datawidth': axi_datawidth})
        print("\t%f [s]" % (time.process_time() - start_time))
        print('# IP-XACT was generated. Check the current directory.')


    skip_export = False
    param_filename = os.path.join(base_dir, '../params/flattened_params/flattened_params.npz')
    chunk_size = 64 # should not be changed
    if not(skip_export):
        # Save the quantized weights
        print("saving params...")
        # convert weight values to a memory image:
        # on a real FPGA platform, this image will be used as a part of the model definition.
        start_time = time.process_time()
        param_data = ng.export_ndarray(output_layers, chunk_size)
        np.savez_compressed(param_filename, param_data)
        print('# weights was saved at %s' % param_filename)
        print("\t%f [s]" % (time.process_time() - start_time))

    input_names = ["reference_image", "hidden_state", "cell_state"]
    for name, layer in zip(input_names, input_layers):
        print("%20s: %6d," % (name, layer.addr), layer.aligned_shape)

    for extern in externs:
        print(extern[-1])
        print("\toutput: addr %d, shape" % extern[0].addr, extern[0].shape, ", aligned_shape", extern[0].aligned_shape)
        for i, e in enumerate(extern[1]):
            print("\tinput%d: addr %d, shape" % (i, e.addr), e.shape, ", aligned_shape", e.aligned_shape)
