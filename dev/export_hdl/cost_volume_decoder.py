import numpy as np
import nngen as ng
from utils import sigmoid, interpolate

def cost_volume_decoder(act0, act81, act86, act91, act96, act106, params, par_ich, par_ochs, par,
                        weight_dtype, bias_dtype, scale_dtype, act_dtype, mid_dtype):

    externs = []

    # [107] interpolate
    act107 = ng.extern([act106], shape=(1, 4, 6, 512), opcode=0x107, func=interpolate(4, 6, 0, "bilinear"))
    externs.append((act107, [act106], "act107 = interpolate(4, 6, 0, 'bilinear')(act106)"))


    # [108] conv
    weight108 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 512), name="decoder_block1.up_convolution.conv.0.weight")
    weight108.set_value(params["decoder_block1.up_convolution.conv.0.weight"])

    bias108 = ng.variable(dtype=bias_dtype, shape=(256,), name="decoder_block1.up_convolution.conv.0.bias")
    bias108.set_value(np.round(params["decoder_block1.up_convolution.conv.0.bias"] / (float) (1 << 3)).astype(params["decoder_block1.up_convolution.conv.0.bias"].dtype))

    scale108 = ng.variable(dtype=scale_dtype, shape=(256,), name="decoder_block1.up_convolution.conv.0.scale")
    scale108.set_value(params["decoder_block1.up_convolution.conv.0.scale"])

    rshift108 = ng.constant([16], dtype=ng.int8)
    act108 = ng.conv2d(act107, weight108, strides=(1, 1, 1, 1), bias=bias108, scale=scale108, rshift_out=rshift108, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [109] cat
    rshift109 = ng.constant([1], dtype=ng.int8)
    act109 = ng.concat([ng.rshift_round(act108, rshift109, par=par), act96], axis=3)


    # [110] conv
    weight110 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 512), name="decoder_block1.convolution1.0.weight")
    weight110.set_value(params["decoder_block1.convolution1.0.weight"])

    bias110 = ng.variable(dtype=bias_dtype, shape=(256,), name="decoder_block1.convolution1.0.bias")
    bias110.set_value(np.round(params["decoder_block1.convolution1.0.bias"] / (float) (1 << 5)).astype(params["decoder_block1.convolution1.0.bias"].dtype))

    scale110 = ng.variable(dtype=scale_dtype, shape=(256,), name="decoder_block1.convolution1.0.scale")
    scale110.set_value(params["decoder_block1.convolution1.0.scale"])

    rshift110 = ng.constant([12], dtype=ng.int8)
    act110 = ng.conv2d(act109, weight110, strides=(1, 1, 1, 1), bias=bias110, scale=scale110, rshift_out=rshift110, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [111] conv
    weight111 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 256), name="decoder_block1.convolution2.0.weight")
    weight111.set_value(params["decoder_block1.convolution2.0.weight"])

    bias111 = ng.variable(dtype=bias_dtype, shape=(256,), name="decoder_block1.convolution2.0.bias")
    bias111.set_value(np.round(params["decoder_block1.convolution2.0.bias"] / (float) (1 << 4)).astype(params["decoder_block1.convolution2.0.bias"].dtype))

    scale111 = ng.variable(dtype=scale_dtype, shape=(256,), name="decoder_block1.convolution2.0.scale")
    scale111.set_value(params["decoder_block1.convolution2.0.scale"])

    rshift111 = ng.constant([11], dtype=ng.int8)
    act111 = ng.conv2d(act110, weight111, strides=(1, 1, 1, 1), bias=bias111, scale=scale111, rshift_out=rshift111, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [112] conv
    weight112 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 256), name="depth_layer_one_sixteen.0.weight")
    weight112.set_value(params["depth_layer_one_sixteen.0.weight"])

    bias112 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_one_sixteen.0.bias")
    bias112.set_value(np.round(params["depth_layer_one_sixteen.0.bias"] / (float) (1 << 7)).astype(params["depth_layer_one_sixteen.0.bias"].dtype))

    rshift112 = ng.constant([18], dtype=ng.int8)
    act112 = ng.conv2d(act111, weight112, strides=(1, 1, 1, 1), bias=bias112, rshift_out=rshift112, act_func=sigmoid, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [113] interpolate
    act113 = ng.extern([act111], shape=(1, 8, 12, 256), opcode=0x113, func=interpolate(8, 12, 0, "bilinear"))
    externs.append((act113, [act111], "act113 = interpolate(8, 12, 0, 'bilinear')(act111)"))


    # [114] conv
    weight114 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 256), name="decoder_block2.up_convolution.conv.0.weight")
    weight114.set_value(params["decoder_block2.up_convolution.conv.0.weight"])

    bias114 = ng.variable(dtype=bias_dtype, shape=(128,), name="decoder_block2.up_convolution.conv.0.bias")
    bias114.set_value(np.round(params["decoder_block2.up_convolution.conv.0.bias"] / (float) (1 << 2)).astype(params["decoder_block2.up_convolution.conv.0.bias"].dtype))

    scale114 = ng.variable(dtype=scale_dtype, shape=(128,), name="decoder_block2.up_convolution.conv.0.scale")
    scale114.set_value(params["decoder_block2.up_convolution.conv.0.scale"])

    rshift114 = ng.constant([13], dtype=ng.int8)
    act114 = ng.conv2d(act113, weight114, strides=(1, 1, 1, 1), bias=bias114, scale=scale114, rshift_out=rshift114, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [115] interpolate
    act115 = ng.extern([act112], shape=(1, 8, 12, 1), opcode=0x115, func=interpolate(8, 12, 0, "bilinear"))
    externs.append((act115, [act112], "act115 = interpolate(8, 12, 0, 'bilinear')(act112)"))


    # [116] cat
    rshift116 = ng.constant([2], dtype=ng.int8)
    act116 = ng.concat([act114, act91, ng.rshift_round(act115, rshift116, par=par)], axis=3)


    # [117] conv
    weight117 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 257), name="decoder_block2.convolution1.0.weight")
    weight117.set_value(params["decoder_block2.convolution1.0.weight"])

    bias117 = ng.variable(dtype=bias_dtype, shape=(128,), name="decoder_block2.convolution1.0.bias")
    bias117.set_value(np.round(params["decoder_block2.convolution1.0.bias"] / (float) (1 << 3)).astype(params["decoder_block2.convolution1.0.bias"].dtype))

    scale117 = ng.variable(dtype=scale_dtype, shape=(128,), name="decoder_block2.convolution1.0.scale")
    scale117.set_value(params["decoder_block2.convolution1.0.scale"])

    rshift117 = ng.constant([15], dtype=ng.int8)
    act117 = ng.conv2d(act116, weight117, strides=(1, 1, 1, 1), bias=bias117, scale=scale117, rshift_out=rshift117, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [118] conv
    weight118 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 128), name="decoder_block2.convolution2.0.weight")
    weight118.set_value(params["decoder_block2.convolution2.0.weight"])

    bias118 = ng.variable(dtype=bias_dtype, shape=(128,), name="decoder_block2.convolution2.0.bias")
    bias118.set_value(np.round(params["decoder_block2.convolution2.0.bias"] / (float) (1 << 5)).astype(params["decoder_block2.convolution2.0.bias"].dtype))

    scale118 = ng.variable(dtype=scale_dtype, shape=(128,), name="decoder_block2.convolution2.0.scale")
    scale118.set_value(params["decoder_block2.convolution2.0.scale"])

    rshift118 = ng.constant([11], dtype=ng.int8)
    act118 = ng.conv2d(act117, weight118, strides=(1, 1, 1, 1), bias=bias118, scale=scale118, rshift_out=rshift118, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [119] conv
    weight119 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 128), name="depth_layer_one_eight.0.weight")
    weight119.set_value(params["depth_layer_one_eight.0.weight"])

    bias119 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_one_eight.0.bias")
    bias119.set_value(np.round(params["depth_layer_one_eight.0.bias"] / (float) (1 << 7)).astype(params["depth_layer_one_eight.0.bias"].dtype))

    rshift119 = ng.constant([17], dtype=ng.int8)
    act119 = ng.conv2d(act118, weight119, strides=(1, 1, 1, 1), bias=bias119, rshift_out=rshift119, act_func=sigmoid, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [120] interpolate
    act120 = ng.extern([act118], shape=(1, 16, 24, 128), opcode=0x120, func=interpolate(16, 24, 0, "bilinear"))
    externs.append((act120, [act118], "act120 = interpolate(16, 24, 0, 'bilinear')(act118)"))


    # [121] conv
    weight121 = ng.variable(dtype=weight_dtype, shape=(64, 3, 3, 128), name="decoder_block3.up_convolution.conv.0.weight")
    weight121.set_value(params["decoder_block3.up_convolution.conv.0.weight"])

    bias121 = ng.variable(dtype=bias_dtype, shape=(64,), name="decoder_block3.up_convolution.conv.0.bias")
    bias121.set_value(np.round(params["decoder_block3.up_convolution.conv.0.bias"] / (float) (1 << 4)).astype(params["decoder_block3.up_convolution.conv.0.bias"].dtype))

    scale121 = ng.variable(dtype=scale_dtype, shape=(64,), name="decoder_block3.up_convolution.conv.0.scale")
    scale121.set_value(params["decoder_block3.up_convolution.conv.0.scale"])

    rshift121 = ng.constant([13], dtype=ng.int8)
    act121 = ng.conv2d(act120, weight121, strides=(1, 1, 1, 1), bias=bias121, scale=scale121, rshift_out=rshift121, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [122] interpolate
    act122 = ng.extern([act119], shape=(1, 16, 24, 1), opcode=0x122, func=interpolate(16, 24, 0, "bilinear"))
    externs.append((act122, [act119], "act122 = interpolate(16, 24, 0, 'bilinear')(act119)"))


    # [123] cat
    rshift123 = ng.constant([3], dtype=ng.int8)
    act123 = ng.concat([act121, act86, ng.rshift_round(act122, rshift123, par=par)], axis=3)


    # [124] conv
    weight124 = ng.variable(dtype=weight_dtype, shape=(64, 3, 3, 129), name="decoder_block3.convolution1.0.weight")
    weight124.set_value(params["decoder_block3.convolution1.0.weight"])

    bias124 = ng.variable(dtype=bias_dtype, shape=(64,), name="decoder_block3.convolution1.0.bias")
    bias124.set_value(np.round(params["decoder_block3.convolution1.0.bias"] / (float) (1 << 3)).astype(params["decoder_block3.convolution1.0.bias"].dtype))

    scale124 = ng.variable(dtype=scale_dtype, shape=(64,), name="decoder_block3.convolution1.0.scale")
    scale124.set_value(params["decoder_block3.convolution1.0.scale"])

    rshift124 = ng.constant([14], dtype=ng.int8)
    act124 = ng.conv2d(act123, weight124, strides=(1, 1, 1, 1), bias=bias124, scale=scale124, rshift_out=rshift124, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [125] conv
    weight125 = ng.variable(dtype=weight_dtype, shape=(64, 3, 3, 64), name="decoder_block3.convolution2.0.weight")
    weight125.set_value(params["decoder_block3.convolution2.0.weight"])

    bias125 = ng.variable(dtype=bias_dtype, shape=(64,), name="decoder_block3.convolution2.0.bias")
    bias125.set_value(np.round(params["decoder_block3.convolution2.0.bias"] / (float) (1 << 4)).astype(params["decoder_block3.convolution2.0.bias"].dtype))

    scale125 = ng.variable(dtype=scale_dtype, shape=(64,), name="decoder_block3.convolution2.0.scale")
    scale125.set_value(params["decoder_block3.convolution2.0.scale"])

    rshift125 = ng.constant([13], dtype=ng.int8)
    act125 = ng.conv2d(act124, weight125, strides=(1, 1, 1, 1), bias=bias125, scale=scale125, rshift_out=rshift125, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [126] conv
    weight126 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 64), name="depth_layer_quarter.0.weight")
    weight126.set_value(params["depth_layer_quarter.0.weight"])

    bias126 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_quarter.0.bias")
    bias126.set_value(np.round(params["depth_layer_quarter.0.bias"] / (float) (1 << 5)).astype(params["depth_layer_quarter.0.bias"].dtype))

    rshift126 = ng.constant([19], dtype=ng.int8)
    act126 = ng.conv2d(act125, weight126, strides=(1, 1, 1, 1), bias=bias126, rshift_out=rshift126, act_func=sigmoid, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [127] interpolate
    act127 = ng.extern([act125], shape=(1, 32, 48, 64), opcode=0x127, func=interpolate(32, 48, 0, "bilinear"))
    externs.append((act127, [act125], "act127 = interpolate(32, 48, 0, 'bilinear')(act125)"))


    # [128] conv
    weight128 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 64), name="decoder_block4.up_convolution.conv.0.weight")
    weight128.set_value(params["decoder_block4.up_convolution.conv.0.weight"])

    bias128 = ng.variable(dtype=bias_dtype, shape=(32,), name="decoder_block4.up_convolution.conv.0.bias")
    bias128.set_value(np.round(params["decoder_block4.up_convolution.conv.0.bias"] / (float) (1 << 2)).astype(params["decoder_block4.up_convolution.conv.0.bias"].dtype))

    scale128 = ng.variable(dtype=scale_dtype, shape=(32,), name="decoder_block4.up_convolution.conv.0.scale")
    scale128.set_value(params["decoder_block4.up_convolution.conv.0.scale"])

    rshift128 = ng.constant([15], dtype=ng.int8)
    act128 = ng.conv2d(act127, weight128, strides=(1, 1, 1, 1), bias=bias128, scale=scale128, rshift_out=rshift128, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(5, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [129] interpolate
    act129 = ng.extern([act126], shape=(1, 32, 48, 1), opcode=0x129, func=interpolate(32, 48, 0, "bilinear"))
    externs.append((act129, [act126], "act129 = interpolate(32, 48, 0, 'bilinear')(act126)"))


    # [130] cat
    rshift130 = ng.constant([3], dtype=ng.int8)
    act130 = ng.concat([act128, act81, ng.rshift_round(act129, rshift130, par=par)], axis=3)


    # [131] conv
    weight131 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 65), name="decoder_block4.convolution1.0.weight")
    weight131.set_value(params["decoder_block4.convolution1.0.weight"])

    bias131 = ng.variable(dtype=bias_dtype, shape=(32,), name="decoder_block4.convolution1.0.bias")
    bias131.set_value(np.round(params["decoder_block4.convolution1.0.bias"] / (float) (1 << 3)).astype(params["decoder_block4.convolution1.0.bias"].dtype))

    scale131 = ng.variable(dtype=scale_dtype, shape=(32,), name="decoder_block4.convolution1.0.scale")
    scale131.set_value(params["decoder_block4.convolution1.0.scale"])

    rshift131 = ng.constant([14], dtype=ng.int8)
    act131 = ng.conv2d(act130, weight131, strides=(1, 1, 1, 1), bias=bias131, scale=scale131, rshift_out=rshift131, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(5, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [132] conv
    weight132 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 32), name="decoder_block4.convolution2.0.weight")
    weight132.set_value(params["decoder_block4.convolution2.0.weight"])

    bias132 = ng.variable(dtype=bias_dtype, shape=(32,), name="decoder_block4.convolution2.0.bias")
    bias132.set_value(np.round(params["decoder_block4.convolution2.0.bias"] / (float) (1 << 4)).astype(params["decoder_block4.convolution2.0.bias"].dtype))

    scale132 = ng.variable(dtype=scale_dtype, shape=(32,), name="decoder_block4.convolution2.0.scale")
    scale132.set_value(params["decoder_block4.convolution2.0.scale"])

    rshift132 = ng.constant([13], dtype=ng.int8)
    act132 = ng.conv2d(act131, weight132, strides=(1, 1, 1, 1), bias=bias132, scale=scale132, rshift_out=rshift132, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(5, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [133] conv
    weight133 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 32), name="depth_layer_half.0.weight")
    weight133.set_value(params["depth_layer_half.0.weight"])

    bias133 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_half.0.bias")
    bias133.set_value(np.round(params["depth_layer_half.0.bias"] / (float) (1 << 7)).astype(params["depth_layer_half.0.bias"].dtype))

    rshift133 = ng.constant([18], dtype=ng.int8)
    act133 = ng.conv2d(act132, weight133, strides=(1, 1, 1, 1), bias=bias133, rshift_out=rshift133, act_func=sigmoid, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [134] interpolate
    act134 = ng.extern([act133], shape=(1, 64, 96, 1), opcode=0x134, func=interpolate(64, 96, 0, "bilinear"))
    externs.append((act134, [act133], "act134 = interpolate(64, 96, 0, 'bilinear')(act133)"))


    # [135] interpolate
    act135 = ng.extern([act132], shape=(1, 64, 96, 32), opcode=0x135, func=interpolate(64, 96, 0, "bilinear"))
    externs.append((act135, [act132], "act135 = interpolate(64, 96, 0, 'bilinear')(act132)"))


    # [136] cat
    rshift136s = [ng.constant([2], dtype=ng.int8), ng.constant([4], dtype=ng.int8)]
    act136 = ng.concat([ng.rshift_round(act135, rshift136s[0], par=par), ng.rshift_round(act134, rshift136s[1], par=par), act0], axis=3)


    # [137] conv
    weight137 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 36), name="refine.0.0.weight")
    weight137.set_value(params["refine.0.0.weight"])

    bias137 = ng.variable(dtype=bias_dtype, shape=(32,), name="refine.0.0.bias")
    bias137.set_value(np.round(params["refine.0.0.bias"] / (float) (1 << 5)).astype(params["refine.0.0.bias"].dtype))

    scale137 = ng.variable(dtype=scale_dtype, shape=(32,), name="refine.0.0.scale")
    scale137.set_value(params["refine.0.0.scale"])

    rshift137 = ng.constant([12], dtype=ng.int8)
    act137 = ng.conv2d(act136, weight137, strides=(1, 1, 1, 1), bias=bias137, scale=scale137, rshift_out=rshift137, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(5, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [138] conv
    weight138 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 32), name="refine.1.0.weight")
    weight138.set_value(params["refine.1.0.weight"])

    bias138 = ng.variable(dtype=bias_dtype, shape=(32,), name="refine.1.0.bias")
    bias138.set_value(np.round(params["refine.1.0.bias"] / (float) (1 << 3)).astype(params["refine.1.0.bias"].dtype))

    scale138 = ng.variable(dtype=scale_dtype, shape=(32,), name="refine.1.0.scale")
    scale138.set_value(params["refine.1.0.scale"])

    rshift138 = ng.constant([13], dtype=ng.int8)
    act138 = ng.conv2d(act137, weight138, strides=(1, 1, 1, 1), bias=bias138, scale=scale138, rshift_out=rshift138, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(5, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [139] conv
    weight139 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 32), name="depth_layer_full.0.weight")
    weight139.set_value(params["depth_layer_full.0.weight"])

    bias139 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_full.0.bias")
    bias139.set_value(np.round(params["depth_layer_full.0.bias"] / (float) (1 << 6)).astype(params["depth_layer_full.0.bias"].dtype))

    rshift139 = ng.constant([18], dtype=ng.int8)
    act139 = ng.conv2d(act138, weight139, strides=(1, 1, 1, 1), bias=bias139, rshift_out=rshift139, act_func=sigmoid, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    return (act139,), externs
