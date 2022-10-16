import numpy as np
import nngen as ng

def cost_volume_encoder(act78, act74, act70, act66, act79, params, par_ich, par_ochs, par,
                        weight_dtype, bias_dtype, scale_dtype, act_dtype, mid_dtype):

    # [80] cat
    rshift80 = ng.constant([2], dtype=ng.int8)
    act80 = ng.concat([ng.rshift_round(act78, rshift80, par=par), act79], axis=3)


    # [81] conv
    weight81 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 96), name="aggregator0.0.weight")
    weight81.set_value(params["aggregator0.0.weight"])

    bias81 = ng.variable(dtype=bias_dtype, shape=(32,), name="aggregator0.0.bias")
    bias81.set_value(np.round(params["aggregator0.0.bias"] / (float) (1 << 5)).astype(params["aggregator0.0.bias"].dtype))

    scale81 = ng.variable(dtype=scale_dtype, shape=(32,), name="aggregator0.0.scale")
    scale81.set_value(params["aggregator0.0.scale"])

    rshift81 = ng.constant([13], dtype=ng.int8)
    act81 = ng.conv2d(act80, weight81, strides=(1, 1, 1, 1), bias=bias81, scale=scale81, rshift_out=rshift81, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(5, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [82] conv
    weight82 = ng.variable(dtype=weight_dtype, shape=(64, 5, 5, 32), name="encoder_block0.down_convolution.down_conv.0.weight")
    weight82.set_value(params["encoder_block0.down_convolution.down_conv.0.weight"])

    bias82 = ng.variable(dtype=bias_dtype, shape=(64,), name="encoder_block0.down_convolution.down_conv.0.bias")
    bias82.set_value(np.round(params["encoder_block0.down_convolution.down_conv.0.bias"] / (float) (1 << 4)).astype(params["encoder_block0.down_convolution.down_conv.0.bias"].dtype))

    scale82 = ng.variable(dtype=scale_dtype, shape=(64,), name="encoder_block0.down_convolution.down_conv.0.scale")
    scale82.set_value(params["encoder_block0.down_convolution.down_conv.0.scale"])

    rshift82 = ng.constant([14], dtype=ng.int8)
    act82 = ng.conv2d(act81, weight82, strides=(1, 2, 2, 1), bias=bias82, scale=scale82, rshift_out=rshift82, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(5, 2)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [83] conv
    weight83 = ng.variable(dtype=weight_dtype, shape=(64, 5, 5, 64), name="encoder_block0.standard_convolution.conv1.0.weight")
    weight83.set_value(params["encoder_block0.standard_convolution.conv1.0.weight"])

    bias83 = ng.variable(dtype=bias_dtype, shape=(64,), name="encoder_block0.standard_convolution.conv1.0.bias")
    bias83.set_value(np.round(params["encoder_block0.standard_convolution.conv1.0.bias"] / (float) (1 << 3)).astype(params["encoder_block0.standard_convolution.conv1.0.bias"].dtype))

    scale83 = ng.variable(dtype=scale_dtype, shape=(64,), name="encoder_block0.standard_convolution.conv1.0.scale")
    scale83.set_value(params["encoder_block0.standard_convolution.conv1.0.scale"])

    rshift83 = ng.constant([15], dtype=ng.int8)
    act83 = ng.conv2d(act82, weight83, strides=(1, 1, 1, 1), bias=bias83, scale=scale83, rshift_out=rshift83, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(5, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [84] conv
    weight84 = ng.variable(dtype=weight_dtype, shape=(64, 5, 5, 64), name="encoder_block0.standard_convolution.conv2.0.weight")
    weight84.set_value(params["encoder_block0.standard_convolution.conv2.0.weight"])

    bias84 = ng.variable(dtype=bias_dtype, shape=(64,), name="encoder_block0.standard_convolution.conv2.0.bias")
    bias84.set_value(np.round(params["encoder_block0.standard_convolution.conv2.0.bias"] / (float) (1 << 3)).astype(params["encoder_block0.standard_convolution.conv2.0.bias"].dtype))

    scale84 = ng.variable(dtype=scale_dtype, shape=(64,), name="encoder_block0.standard_convolution.conv2.0.scale")
    scale84.set_value(params["encoder_block0.standard_convolution.conv2.0.scale"])

    rshift84 = ng.constant([15], dtype=ng.int8)
    act84 = ng.conv2d(act83, weight84, strides=(1, 1, 1, 1), bias=bias84, scale=scale84, rshift_out=rshift84, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(5, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [85] cat
    rshift85 = ng.constant([3], dtype=ng.int8)
    act85 = ng.concat([act74, ng.rshift_round(act84, rshift85, par=par)], axis=3)


    # [86] conv
    weight86 = ng.variable(dtype=weight_dtype, shape=(64, 3, 3, 96), name="aggregator1.0.weight")
    weight86.set_value(params["aggregator1.0.weight"])

    bias86 = ng.variable(dtype=bias_dtype, shape=(64,), name="aggregator1.0.bias")
    bias86.set_value(np.round(params["aggregator1.0.bias"] / (float) (1 << 6)).astype(params["aggregator1.0.bias"].dtype))

    scale86 = ng.variable(dtype=scale_dtype, shape=(64,), name="aggregator1.0.scale")
    scale86.set_value(params["aggregator1.0.scale"])

    rshift86 = ng.constant([13], dtype=ng.int8)
    act86 = ng.conv2d(act85, weight86, strides=(1, 1, 1, 1), bias=bias86, scale=scale86, rshift_out=rshift86, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [87] conv
    weight87 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 64), name="encoder_block1.down_convolution.down_conv.0.weight")
    weight87.set_value(params["encoder_block1.down_convolution.down_conv.0.weight"])

    bias87 = ng.variable(dtype=bias_dtype, shape=(128,), name="encoder_block1.down_convolution.down_conv.0.bias")
    bias87.set_value(np.round(params["encoder_block1.down_convolution.down_conv.0.bias"] / (float) (1 << 5)).astype(params["encoder_block1.down_convolution.down_conv.0.bias"].dtype))

    scale87 = ng.variable(dtype=scale_dtype, shape=(128,), name="encoder_block1.down_convolution.down_conv.0.scale")
    scale87.set_value(params["encoder_block1.down_convolution.down_conv.0.scale"])

    rshift87 = ng.constant([13], dtype=ng.int8)
    act87 = ng.conv2d(act86, weight87, strides=(1, 2, 2, 1), bias=bias87, scale=scale87, rshift_out=rshift87, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 2)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [88] conv
    weight88 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 128), name="encoder_block1.standard_convolution.conv1.0.weight")
    weight88.set_value(params["encoder_block1.standard_convolution.conv1.0.weight"])

    bias88 = ng.variable(dtype=bias_dtype, shape=(128,), name="encoder_block1.standard_convolution.conv1.0.bias")
    bias88.set_value(np.round(params["encoder_block1.standard_convolution.conv1.0.bias"] / (float) (1 << 4)).astype(params["encoder_block1.standard_convolution.conv1.0.bias"].dtype))

    scale88 = ng.variable(dtype=scale_dtype, shape=(128,), name="encoder_block1.standard_convolution.conv1.0.scale")
    scale88.set_value(params["encoder_block1.standard_convolution.conv1.0.scale"])

    rshift88 = ng.constant([14], dtype=ng.int8)
    act88 = ng.conv2d(act87, weight88, strides=(1, 1, 1, 1), bias=bias88, scale=scale88, rshift_out=rshift88, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [89] conv
    weight89 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 128), name="encoder_block1.standard_convolution.conv2.0.weight")
    weight89.set_value(params["encoder_block1.standard_convolution.conv2.0.weight"])

    bias89 = ng.variable(dtype=bias_dtype, shape=(128,), name="encoder_block1.standard_convolution.conv2.0.bias")
    bias89.set_value(np.round(params["encoder_block1.standard_convolution.conv2.0.bias"] / (float) (1 << 3)).astype(params["encoder_block1.standard_convolution.conv2.0.bias"].dtype))

    scale89 = ng.variable(dtype=scale_dtype, shape=(128,), name="encoder_block1.standard_convolution.conv2.0.scale")
    scale89.set_value(params["encoder_block1.standard_convolution.conv2.0.scale"])

    rshift89 = ng.constant([15], dtype=ng.int8)
    act89 = ng.conv2d(act88, weight89, strides=(1, 1, 1, 1), bias=bias89, scale=scale89, rshift_out=rshift89, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [90] cat
    rshift90 = ng.constant([1], dtype=ng.int8)
    act90 = ng.concat([act70, ng.rshift_round(act89, rshift90, par=par)], axis=3)


    # [91] conv
    weight91 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 160), name="aggregator2.0.weight")
    weight91.set_value(params["aggregator2.0.weight"])

    bias91 = ng.variable(dtype=bias_dtype, shape=(128,), name="aggregator2.0.bias")
    bias91.set_value(np.round(params["aggregator2.0.bias"] / (float) (1 << 5)).astype(params["aggregator2.0.bias"].dtype))

    scale91 = ng.variable(dtype=scale_dtype, shape=(128,), name="aggregator2.0.scale")
    scale91.set_value(params["aggregator2.0.scale"])

    rshift91 = ng.constant([14], dtype=ng.int8)
    act91 = ng.conv2d(act90, weight91, strides=(1, 1, 1, 1), bias=bias91, scale=scale91, rshift_out=rshift91, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [92] conv
    weight92 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 128), name="encoder_block2.down_convolution.down_conv.0.weight")
    weight92.set_value(params["encoder_block2.down_convolution.down_conv.0.weight"])

    bias92 = ng.variable(dtype=bias_dtype, shape=(256,), name="encoder_block2.down_convolution.down_conv.0.bias")
    bias92.set_value(np.round(params["encoder_block2.down_convolution.down_conv.0.bias"] / (float) (1 << 4)).astype(params["encoder_block2.down_convolution.down_conv.0.bias"].dtype))

    scale92 = ng.variable(dtype=scale_dtype, shape=(256,), name="encoder_block2.down_convolution.down_conv.0.scale")
    scale92.set_value(params["encoder_block2.down_convolution.down_conv.0.scale"])

    rshift92 = ng.constant([15], dtype=ng.int8)
    act92 = ng.conv2d(act91, weight92, strides=(1, 2, 2, 1), bias=bias92, scale=scale92, rshift_out=rshift92, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 2)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [93] conv
    weight93 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 256), name="encoder_block2.standard_convolution.conv1.0.weight")
    weight93.set_value(params["encoder_block2.standard_convolution.conv1.0.weight"])

    bias93 = ng.variable(dtype=bias_dtype, shape=(256,), name="encoder_block2.standard_convolution.conv1.0.bias")
    bias93.set_value(np.round(params["encoder_block2.standard_convolution.conv1.0.bias"] / (float) (1 << 4)).astype(params["encoder_block2.standard_convolution.conv1.0.bias"].dtype))

    scale93 = ng.variable(dtype=scale_dtype, shape=(256,), name="encoder_block2.standard_convolution.conv1.0.scale")
    scale93.set_value(params["encoder_block2.standard_convolution.conv1.0.scale"])

    rshift93 = ng.constant([14], dtype=ng.int8)
    act93 = ng.conv2d(act92, weight93, strides=(1, 1, 1, 1), bias=bias93, scale=scale93, rshift_out=rshift93, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [94] conv
    weight94 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 256), name="encoder_block2.standard_convolution.conv2.0.weight")
    weight94.set_value(params["encoder_block2.standard_convolution.conv2.0.weight"])

    bias94 = ng.variable(dtype=bias_dtype, shape=(256,), name="encoder_block2.standard_convolution.conv2.0.bias")
    bias94.set_value(np.round(params["encoder_block2.standard_convolution.conv2.0.bias"] / (float) (1 << 3)).astype(params["encoder_block2.standard_convolution.conv2.0.bias"].dtype))

    scale94 = ng.variable(dtype=scale_dtype, shape=(256,), name="encoder_block2.standard_convolution.conv2.0.scale")
    scale94.set_value(params["encoder_block2.standard_convolution.conv2.0.scale"])

    rshift94 = ng.constant([15], dtype=ng.int8)
    act94 = ng.conv2d(act93, weight94, strides=(1, 1, 1, 1), bias=bias94, scale=scale94, rshift_out=rshift94, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [95] cat
    rshift95 = ng.constant([1], dtype=ng.int8)
    act95 = ng.concat([act66, ng.rshift_round(act94, rshift95, par=par)], axis=3)


    # [96] conv
    weight96 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 288), name="aggregator3.0.weight")
    weight96.set_value(params["aggregator3.0.weight"])

    bias96 = ng.variable(dtype=bias_dtype, shape=(256,), name="aggregator3.0.bias")
    bias96.set_value(np.round(params["aggregator3.0.bias"] / (float) (1 << 5)).astype(params["aggregator3.0.bias"].dtype))

    scale96 = ng.variable(dtype=scale_dtype, shape=(256,), name="aggregator3.0.scale")
    scale96.set_value(params["aggregator3.0.scale"])

    rshift96 = ng.constant([13], dtype=ng.int8)
    act96 = ng.conv2d(act95, weight96, strides=(1, 1, 1, 1), bias=bias96, scale=scale96, rshift_out=rshift96, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [97] conv
    weight97 = ng.variable(dtype=weight_dtype, shape=(512, 3, 3, 256), name="encoder_block3.down_convolution.down_conv.0.weight")
    weight97.set_value(params["encoder_block3.down_convolution.down_conv.0.weight"])

    bias97 = ng.variable(dtype=bias_dtype, shape=(512,), name="encoder_block3.down_convolution.down_conv.0.bias")
    bias97.set_value(np.round(params["encoder_block3.down_convolution.down_conv.0.bias"] / (float) (1 << 4)).astype(params["encoder_block3.down_convolution.down_conv.0.bias"].dtype))

    scale97 = ng.variable(dtype=scale_dtype, shape=(512,), name="encoder_block3.down_convolution.down_conv.0.scale")
    scale97.set_value(params["encoder_block3.down_convolution.down_conv.0.scale"])

    rshift97 = ng.constant([15], dtype=ng.int8)
    act97 = ng.conv2d(act96, weight97, strides=(1, 2, 2, 1), bias=bias97, scale=scale97, rshift_out=rshift97, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 2)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [98] conv
    weight98 = ng.variable(dtype=weight_dtype, shape=(512, 3, 3, 512), name="encoder_block3.standard_convolution.conv1.0.weight")
    weight98.set_value(params["encoder_block3.standard_convolution.conv1.0.weight"])

    bias98 = ng.variable(dtype=bias_dtype, shape=(512,), name="encoder_block3.standard_convolution.conv1.0.bias")
    bias98.set_value(np.round(params["encoder_block3.standard_convolution.conv1.0.bias"] / (float) (1 << 3)).astype(params["encoder_block3.standard_convolution.conv1.0.bias"].dtype))

    scale98 = ng.variable(dtype=scale_dtype, shape=(512,), name="encoder_block3.standard_convolution.conv1.0.scale")
    scale98.set_value(params["encoder_block3.standard_convolution.conv1.0.scale"])

    rshift98 = ng.constant([14], dtype=ng.int8)
    act98 = ng.conv2d(act97, weight98, strides=(1, 1, 1, 1), bias=bias98, scale=scale98, rshift_out=rshift98, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [99] conv
    weight99 = ng.variable(dtype=weight_dtype, shape=(512, 3, 3, 512), name="encoder_block3.standard_convolution.conv2.0.weight")
    weight99.set_value(params["encoder_block3.standard_convolution.conv2.0.weight"])

    bias99 = ng.variable(dtype=bias_dtype, shape=(512,), name="encoder_block3.standard_convolution.conv2.0.bias")
    bias99.set_value(np.round(params["encoder_block3.standard_convolution.conv2.0.bias"] / (float) (1 << 2)).astype(params["encoder_block3.standard_convolution.conv2.0.bias"].dtype))

    scale99 = ng.variable(dtype=scale_dtype, shape=(512,), name="encoder_block3.standard_convolution.conv2.0.scale")
    scale99.set_value(params["encoder_block3.standard_convolution.conv2.0.scale"])

    rshift99 = ng.constant([16], dtype=ng.int8)
    act99 = ng.conv2d(act98, weight99, strides=(1, 1, 1, 1), bias=bias99, scale=scale99, rshift_out=rshift99, act_func=ng.relu, asymmetric_clip=True, par_ich=par_ich, par_och=par_ochs[(3, 1)], dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    return act81, act86, act91, act96, act99
