"""Defines the structure of the proposed hierarchical encoder.

The visual features, extracted from Res4, Res5, and Res6, exactly align with the
per-layer style codes fed into the StyleGAN generator.
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

try:
    from tensorflow.contrib.nccl.python.ops.nccl_ops import _validate_and_load_nccl_so
except Exception:
    pass
else:
    _validate_and_load_nccl_so()
from tensorflow.contrib.nccl.ops import gen_nccl_ops


def get_sync_bn_mean_var(x, axis, num_dev):
    coef = tf.constant(np.float32(1.0 / num_dev), name="coef")
    shared_name = tf.get_variable_scope().name
    shared_name = '_'.join(shared_name.split('/')[-2:])
    with tf.device(x.device):
        batch_mean = tf.reduce_mean(x, axis=axis)
        batch_mean = gen_nccl_ops.nccl_all_reduce(
            input=batch_mean,
            reduction='sum',
            num_devices=num_dev,
            shared_name=shared_name + '_NCCL_mean') * coef
    with tf.device(x.device):
        batch_mean_square = tf.reduce_mean(tf.square(x), axis=axis)
        batch_mean_square = gen_nccl_ops.nccl_all_reduce(
            input=batch_mean_square,
            reduction='sum',
            num_devices=num_dev,
            shared_name=shared_name + '_NCCL_mean_square') * coef

    batch_var = batch_mean_square - tf.square(batch_mean)

    return batch_mean, batch_var


def sync_batch_norm(inputs,
                    is_training,
                    num_dev=8,
                    center=True,
                    scale=True,
                    decay=0.9,
                    epsilon=1e-05,
                    data_format="NCHW",
                    updates_collections=None,
                    scope='batch_norm'):
    if data_format not in {"NCHW", "NHWC"}:
        raise ValueError(
            "Invalid data_format {}. Allowed: NCHW, NHWC.".format(data_format))
    with tf.variable_scope(scope, 'BatchNorm', values=[inputs]):
        inputs = tf.convert_to_tensor(inputs)
        # is_training = tf.cast(is_training, tf.bool)
        original_dtype = inputs.dtype
        original_shape = inputs.get_shape()
        original_input = inputs

        num_channels = inputs.shape[1].value if data_format == 'NCHW' else inputs.shape[-1].value
        if num_channels is None:
            raise ValueError("`C` dimension must be known but is None")

        original_rank = original_shape.ndims
        if original_rank is None:
            raise ValueError("Inputs %s has undefined rank" % inputs.name)
        elif original_rank not in [2, 4]:
            raise ValueError(
                "Inputs %s has unsupported rank."
                " Expected 2 or 4 but got %d" % (inputs.name, original_rank))

        # Bring 2-D inputs into 4-D format.
        if original_rank == 2:
            new_shape = [-1, 1, 1, num_channels]
            if data_format == "NCHW":
                new_shape = [-1, num_channels, 1, 1]
            inputs = tf.reshape(inputs, new_shape)
        input_shape = inputs.get_shape()
        input_rank = input_shape.ndims

        param_shape_broadcast = None
        if data_format == "NCHW":
            param_shape_broadcast = list([1, num_channels] +
                                         [1 for _ in range(2, input_rank)])

        moving_variables_collections = [
            tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
            tf.GraphKeys.MODEL_VARIABLES,
            tf.GraphKeys.GLOBAL_VARIABLES,
        ]
        moving_mean = tf.get_variable(
            "moving_mean",
            shape=[num_channels],
            initializer=tf.zeros_initializer(),
            trainable=False,
            partitioner=None,
            collections=moving_variables_collections)
        moving_variance = tf.get_variable(
            "moving_variance",
            shape=[num_channels],
            initializer=tf.ones_initializer(),
            trainable=False,
            partitioner=None,
            collections=moving_variables_collections)
        moving_vars_fn = lambda: (moving_mean, moving_variance)

        if not is_training:
            mean, variance = moving_vars_fn()
        else:
            axis = 1 if data_format == "NCHW" else 3
            inputs = tf.cast(inputs, tf.float32)
            moments_axes = [i for i in range(4) if i != axis]
            mean, variance = get_sync_bn_mean_var(inputs, axis=moments_axes, num_dev=num_dev)
            mean = tf.reshape(mean, [-1])
            variance = tf.reshape(variance, [-1])
            if updates_collections is None:
                def _force_update():
                    # Update variables for mean and variance during training.
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean,
                        tf.cast(mean, moving_mean.dtype),
                        decay,
                        zero_debias=False)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance,
                        tf.cast(variance, moving_variance.dtype),
                        decay,
                        zero_debias=False)
                    with tf.control_dependencies(
                            [update_moving_mean, update_moving_variance]):
                        return tf.identity(mean), tf.identity(variance)

                mean, variance = _force_update()
            else:
                def _delay_update():
                    # Update variables for mean and variance during training.
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean,
                        tf.cast(mean, moving_mean.dtype),
                        decay,
                        zero_debias=False)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance,
                        tf.cast(variance, moving_variance.dtype),
                        decay,
                        zero_debias=False)
                    return update_moving_mean, update_moving_variance

                update_mean, update_variance = tf.cond(is_training,
                                                       _delay_update, moving_vars_fn)
                tf.add_to_collections(updates_collections, update_mean)
                tf.add_to_collections(updates_collections, update_variance)
                vars_fn = lambda: (mean, variance)
                mean, variance = vars_fn()

        variables_collections = [
            tf.GraphKeys.MODEL_VARIABLES,
            tf.GraphKeys.GLOBAL_VARIABLES,
        ]
        beta, gamma = None, None
        if scale:
            gamma = tf.get_variable(
                'gamma',
                [num_channels],
                collections=variables_collections,
                initializer=tf.ones_initializer())
        if center:
            beta = tf.get_variable(
                'beta',
                [num_channels],
                collections=variables_collections,
                initializer=tf.zeros_initializer())

        if data_format == 'NCHW':
            mean = tf.reshape(mean, param_shape_broadcast)
            variance = tf.reshape(variance, param_shape_broadcast)
            if beta is not None:
                beta = tf.reshape(beta, param_shape_broadcast)
            if gamma is not None:
                gamma = tf.reshape(gamma, param_shape_broadcast)

        outputs = tf.nn.batch_normalization(
            inputs,
            mean=mean,
            variance=variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=epsilon)
        outputs = tf.cast(outputs, original_dtype)

        outputs.set_shape(input_shape)
        # Bring 2-D inputs back into 2-D format.
        if original_rank == 2:
            outputs = tf.reshape(outputs, tf.shape(original_input))

        return outputs


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.random_normal_initializer(stddev=0.02)) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))


def dense(x, fmaps, gain=1., use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)


def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False, stride=1):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1, 1, stride, stride], padding='SAME', data_format='NCHW')


def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.constant_initializer(0.0))
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])


def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

def relu(x):
    with tf.name_scope('Relu'):
        return tf.maximum(0.0, x)


def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, np.int32(s[1]), np.int32(s[2]), 1, np.int32(s[3]), 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, np.int32(s[1]), np.int32(s[2]) * factor, np.int32(s[3]) * factor])
        return x


def shortcut(inputs, fout, stride=2):
    x_shortcut = conv2d(inputs, fmaps=fout, kernel=1, stride=stride)
    return x_shortcut


def basicblock(inputs, num_filter, stride, dim_match, is_training, num_dev, scope):
    hidden = num_filter
    fout = num_filter
    with tf.variable_scope(scope):
        with tf.variable_scope('Shortcut'):
            if not dim_match:
                x_shortcut = shortcut(inputs, fout, stride=stride)
                x_shortcut = sync_batch_norm(x_shortcut, is_training=is_training, num_dev=num_dev)
            else:
                x_shortcut = inputs
        with tf.variable_scope('Conva'):
            net = apply_bias(conv2d(inputs, fmaps=hidden, kernel=3, stride=stride))
            net = leaky_relu(net)
            net = sync_batch_norm(net, is_training=is_training, num_dev=num_dev)
        with tf.variable_scope('Convb'):
            net = apply_bias(conv2d(net, fmaps=fout, kernel=3))
            net = leaky_relu(net)
            net = sync_batch_norm(net, is_training=is_training, num_dev=num_dev)
        net = net + x_shortcut
    return net


def bottleneck(inputs, num_filter, stride, dim_match, is_training, num_dev, scope):
    hidden = int(num_filter*0.25)
    fout = num_filter
    with tf.variable_scope(scope):
        with tf.variable_scope('Shortcut'):
            if not dim_match:
                x_shortcut = shortcut(inputs, fout, stride=stride)
                x_shortcut = sync_batch_norm(x_shortcut, is_training=is_training, num_dev=num_dev)
            else:
                x_shortcut = inputs
        with tf.variable_scope('Conva'):
            net = apply_bias(conv2d(inputs, fmaps=hidden, kernel=1))
            net = sync_batch_norm(net, is_training=is_training, num_dev=num_dev)
            net = relu(net)
        with tf.variable_scope('Convb'):
            net = apply_bias(conv2d(net, fmaps=hidden, kernel=3, stride=stride))
            net = sync_batch_norm(net, is_training=is_training, num_dev=num_dev)
            net = relu(net)
        with tf.variable_scope('Convc'):
            net = apply_bias(conv2d(net, fmaps=fout, kernel=1))
            net = sync_batch_norm(net, is_training=is_training, num_dev=num_dev)
        net = net + x_shortcut
        net = relu(net)
    return net


def residual_unit(inputs, unit_type, num_filter, stride, dim_match, is_training, num_dev, scope):
    if unit_type == 'bottleneck':
        func = bottleneck
    elif unit_type == 'basicblock':
        func = basicblock
    else:
        raise NotImplementedError
    net = func(inputs=inputs, num_filter=num_filter, stride=stride, dim_match=dim_match, is_training=is_training, num_dev=num_dev, scope=scope)
    return net


def resnet_backbone(inputs, unit_type, units, num_stages, filter_list, is_training, num_dev):
    with tf.variable_scope('Conv0'):
        input = inputs
        net = conv2d(inputs, fmaps=filter_list[0], kernel=7, stride=2)
        conv1 = net
        net = sync_batch_norm(net, is_training=is_training, num_dev=num_dev)
        bn1 = net
        net = leaky_relu(net)
        relu1 = net
        net =  tf.nn.max_pool(net, ksize=[1,1,3,3], strides=[1,1,2,2], padding='SAME', data_format='NCHW')
        down1 = net
    res1 = net
    for i in range(num_stages):
        if i == 0:
            stride = 1
        else:
            stride = 2

        net = residual_unit(inputs=net,
                            num_filter=filter_list[i+1],
                            stride=stride,
                            unit_type=unit_type,
                            num_dev=num_dev,
                            is_training=is_training,
                            dim_match=False,
                            scope=f'stage{i+1}_unit{1}',
                            )
        for j in range(units[i]-1):
            net = residual_unit(inputs=net,
                                num_filter=filter_list[i+1],
                                stride=1,
                                unit_type=unit_type,
                                num_dev=num_dev,
                                is_training=is_training,
                                dim_match=True,
                                scope=f'stage{i+1}_unit{j+2}')
        if i == 0:
            res2 = net
        elif i == 1:
            res3 = net
        elif i == 2:
            res4 = net
        else:
            res5 = net

    return input, conv1, bn1, relu1, down1, res1, res2, res3, res4, res5


def get_resnet_backbone(inputs,
                        num_layers=50,
                        is_training=True,
                        num_gpus=8):
    num_stages = 4
    if num_layers == 18:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
        units = [3, 4, 6, 3]
    elif num_layers == 50:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
        units = [3, 4, 6, 3]
    elif num_layers == 101:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
        units = [3, 4, 23, 3]
    else:
        raise NotImplementedError('We do not implement resnet with depth {}'.format(num_layers))

    unit_type = 'basicblock' if not bottle_neck else 'bottleneck'

    input, covn1, bn1, relu1, down1, res1, res2, res3, res4, res5 = resnet_backbone(inputs=inputs,
                                                   unit_type=unit_type,
                                                   units=units,
                                                   num_stages=num_stages,
                                                   filter_list=filter_list,
                                                   is_training=is_training,
                                                   num_dev=num_gpus)
    return input, covn1, bn1, relu1, down1, res1, res2, res3, res4, res5


def fpn(inputs, start_level=3, fout=256, scope='fpn'):
    # len(inputs) == 6 0,1,2,3,4,5
    with tf.variable_scope(scope):
        laterals = []
        for i in range(start_level, len(inputs)):
            linput = inputs[i]
            with tf.variable_scope(f'lconv_{i}'):
                linput = apply_bias(conv2d(linput, fmaps=fout, kernel=3, stride=1))
            laterals.append(linput)

        flevel = len(laterals)
        for i in range(flevel-1, 0, -1):
            laterals[i-1] += upscale2d(laterals[i])

        outs = []
        for i in range(0, flevel):
            finput = laterals[i]
            with tf.variable_scope(f'fconv_{i}'):
                finput = apply_bias(conv2d(finput, fmaps=fout, kernel=3, stride=1))
            outs.append(finput)
    return outs


def sam(inputs, fout):
    # recurrent downsample
    for i in range(len(inputs)-1):
        for j in range(0, len(inputs)-1-i):
            inputs[j] = downscale2d(inputs[j])

    for i in range(len(inputs)):
        with tf.variable_scope(f'fuse_conv_{i}'):
            inputs[i] = apply_bias(conv2d(inputs[i], fmaps=fout, kernel=3, stride=1))
    # latent_fusion
    for i in range(len(inputs)-1):
        inputs[i] = inputs[i] + inputs[-1]

    return inputs


def Encoder(
            Input_img,                 # Input image: [Minibatch, Channel, Height, Width].
            size             = 256,    # Input image size.
            depth            = 18,     # ResNet Depth
            num_layers       = 14,     # Number of layers in in G_synthesis().
            is_training      = True,   # Whether or not the layer is in training mode?
            num_gpus         = 8,      # Number of gpus to use
            dlatent_size     = 512,    # Disentangled latent (W) dimensionality.
            **kwargs):
    # Extract multi-level feature with resnet backbone
    max_length = dlatent_size * 2
    dsize = [max_length] * 8 + [max_length // 2] * 2 + [max_length // 4] * 2 + [max_length // 8] * 2
    Input_img.set_shape([None, 3, size, size])
    input, covn1, bn1, relu1, down1, res1, res2, res3, res4, res5 = get_resnet_backbone(inputs=Input_img,
                                                       num_layers=depth,
                                                       is_training=is_training,
                                                       num_gpus=num_gpus)
    # Add another residual block to obtain lower resolution feature map
    net = residual_unit(inputs=res5,
                        unit_type='bottleneck',
                        num_filter=2048,
                        stride=2,
                        dim_match=False,
                        num_dev=num_gpus,
                        is_training=is_training,
                        scope='stage5')
    res6 = net
    inputs = (res1, res2, res3, res4, res5, res6)

    # FPN and Sam for feature fusion
    inputs = fpn(inputs, start_level=3, fout=512)
    inputs = sam(list(inputs), fout=512)

    # Our paper only consider use res4, res5, res6 to obtain GH-Feat
    assert len(inputs) == 3
    res4, res5, res6 = inputs

    # res6 generate GH-Feat from level 11 to level 14
    with tf.variable_scope('HighLevel'):
        latent_w0 = apply_bias(dense(res6, fmaps=sum(dsize[:4])))
        latent_w0 = sync_batch_norm(latent_w0, is_training=is_training, num_dev=num_gpus)
        latent_w0 = tf.reshape(latent_w0, [-1, 4, dsize[0]])

    # res5 generate GH-Feat from level 7 to level 10
    with tf.variable_scope('MidLevel'):
        latent_w1 = apply_bias(dense(res5, fmaps=sum(dsize[4:8])))
        latent_w1 = sync_batch_norm(latent_w1, is_training=is_training, num_dev=num_gpus)
        latent_w1 = tf.reshape(latent_w1, [-1, 4, dsize[4]])

    # res4 generate GH-Feat from level 1 to level 6
    with tf.variable_scope('LowLevel'):
        latent_w2 = apply_bias(dense(res4, fmaps=sum(dsize[8:])))
        latent_w2 = sync_batch_norm(latent_w2, is_training=is_training, num_dev=num_gpus)
        latent_w20 = tf.tile(tf.reshape(latent_w2[:, :sum(dsize[8:10])], [-1, 2, 512]), [1, 1, max_length//512])
        latent_w21 = tf.tile(tf.reshape(latent_w2[:, sum(dsize[8:10]):sum(dsize[8:12])], [-1, 2, 256]),
                             [1, 1, max_length//256])
        latent_w22 = tf.tile(tf.reshape(latent_w2[:, sum(dsize[8:12]):], [-1, 2, 128]), [1, 1, max_length//128])

    with tf.variable_scope('Latent_out'):
        latent_w = tf.concat([latent_w0, latent_w1, latent_w20, latent_w21, latent_w22], axis=1)

    return latent_w
