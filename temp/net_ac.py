import tensorflow as tf
from hyper_parameters_ac import *


def create_variables(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def variable_with_weight_decay(name, shape, wd):
    var = create_variables(
        name,
        shape,
        tf.glorot_uniform_initializer())
    # tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
    # tf.glorot_uniform_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def batch_normalization_layer(input_layer, train_phase=True):
    bn_layer = tf.layers.batch_normalization(input_layer, training=train_phase, name='bn')
    return bn_layer


def conv_layer(input_layer, filter_shape, stride, padding, wd=None):
    filter_kernel = variable_with_weight_decay(name='weights', shape=filter_shape, wd=wd)
    conv_layer = tf.nn.conv2d(input_layer, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    biases = create_variables('biases', filter_shape[-1], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv_layer, biases)
    return pre_activation


def depthwise_layer(input_layer, filter_shape, stride, padding, wd=None):
    filter_kernel = variable_with_weight_decay(name='weights', shape=filter_shape, wd=wd)
    conv_layer = tf.nn.depthwise_conv2d(input_layer, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    biases = create_variables('biases', filter_shape[-2], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv_layer, biases)
    return pre_activation


def depthwise_bn_relu_layer(input_layer, ks, stride, padding, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    filter_kernel = variable_with_weight_decay(name='weights', shape=[ks, ks, input_channel, 1], wd=wd)
    conv_layer = tf.nn.depthwise_conv2d(input_layer, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    bn_layer = batch_normalization_layer(conv_layer, train_phase=bn)
    output = tf.nn.relu(bn_layer)
    return output


def conv_bn_relu_layer(input_layer, filter_shape, stride, padding, bn=True, wd=None):
    filter_kernel = variable_with_weight_decay(name='weights', shape=filter_shape, wd=wd)
    conv_layer = tf.nn.conv2d(input_layer, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    bn_layer = batch_normalization_layer(conv_layer, train_phase=bn)
    output = tf.nn.relu(bn_layer)
    return output


def steam_block(input_layer, bn=True, wd=None):
    with tf.variable_scope('conv1'):
        conv1_layer = conv_bn_relu_layer(input_layer, [3, 3, 3, 32], 1, padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('conv2_1'):
        conv2_1_layer = conv_bn_relu_layer(conv1_layer, [1, 1, 32, 16], 1, padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('conv2_2'):
        conv2_2_layer = conv_bn_relu_layer(conv2_1_layer, [3, 3, 16, 32], 2, padding='SAME', bn=bn, wd=wd)

    maxpool_layer = tf.nn.max_pool(conv1_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    concat_layer = tf.concat([conv2_2_layer, maxpool_layer], axis=-1)

    with tf.variable_scope('conv3'):
        output_layer = conv_bn_relu_layer(concat_layer, [1, 1, 64, 32], 1, padding='SAME', bn=bn, wd=wd)

    return output_layer


def dense_layer(input_layer, k, bottleneck_width, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    bottle_channel = k * bottleneck_width

    with tf.variable_scope('left_conv1'):
        left_conv1_layer = conv_bn_relu_layer(input_layer, [1, 1, input_channel, bottle_channel], 1,
                                              padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('left_conv2'):
        left_conv2_layer = conv_bn_relu_layer(left_conv1_layer, [3, 3, bottle_channel, k], 1,
                                              padding='SAME', bn=bn, wd=wd)

    output = tf.concat([input_layer, left_conv2_layer], axis=3)
    return output


def dense_layer2(input_layer, k, bottleneck_width, linear_width=1, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    bottle_channel = k * bottleneck_width

    with tf.variable_scope('right_conv1'):
        right_conv1_layer = conv_bn_relu_layer(input_layer, [1, 1, input_channel, bottle_channel], 1,
                                               padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('right_conv2'):
        right_conv2_layer = conv_bn_relu_layer(right_conv1_layer, [3, 3, bottle_channel, linear_width * k], 1,
                                               padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('linear_conv'):
        conv0 = conv_bn_relu_layer(input_layer, [1, 1, input_channel, linear_width * k], 1,
                                   padding='SAME', bn=bn, wd=wd)

    output = tf.concat([conv0, right_conv2_layer], axis=-1)
    return output


def dense_block(input_layer, num_dense_layer, k, bottleneck_width, bn=True, wd=None):
    output = input_layer
    for i in range(num_dense_layer):
        with tf.variable_scope('dense_layer_%d' % (i + 1)):
            output = dense_layer(output, k, bottleneck_width, bn=bn, wd=wd)

    return output


def transition_layer(input_layer, output_channel, is_pool=True, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    with tf.variable_scope('conv1'):
        conv1_layer = conv_bn_relu_layer(input_layer, [1, 1, input_channel, output_channel], 1,
                                         padding='SAME', bn=bn, wd=wd)

    if is_pool:
        output = tf.nn.max_pool(conv1_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    else:
        output = conv1_layer

    return output


def conv_bn_layer(input_layer, filter_shape, stride, padding, bn=True, wd=None):
    filter_kernel = variable_with_weight_decay(name='weights', shape=filter_shape, wd=wd)
    conv_layer = tf.nn.conv2d(input_layer, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    bn_layer = batch_normalization_layer(conv_layer, train_phase=bn)
    return bn_layer


def res_block(input_layer, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    with tf.variable_scope('left_conv1'):
        left_conv1_layer = conv_bn_relu_layer(input_layer, [1, 1, input_channel, 128], 1,
                                              padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('left_conv2'):
        left_conv2_layer = conv_bn_layer(left_conv1_layer, [3, 3, 128, 128], 1,
                                         padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('right_conv1'):
        right_conv1_layer = conv_bn_layer(input_layer, [1, 1, input_channel, 128], 1,
                                          padding='SAME', bn=bn, wd=wd)

    output_add = left_conv2_layer + right_conv1_layer
    output = tf.nn.relu(output_add)
    return output


def dense_block_ext(input_layer, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    with tf.variable_scope('left_conv1'):
        left_conv1_layer = conv_bn_relu_layer(input_layer, [1, 1, input_channel, 128], 1,
                                              padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('left_conv2'):
        left_conv2_layer = conv_bn_relu_layer(left_conv1_layer, [3, 3, 128, 128], 2,
                                              padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('right_conv1'):
        pool_layer = tf.nn.max_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME')
        right_conv1_layer = conv_bn_relu_layer(pool_layer, [1, 1, input_channel, 128], 1,
                                               padding='SAME', bn=bn, wd=wd)

    output = tf.concat([left_conv2_layer, right_conv1_layer], axis=-1)
    return output


def inference(input_tensor_batch, bn, k=32, block_config=BLOCK_CONFIG,
              bottleneck_width=BOTTLENECK_WIDTH, reuse=False):
    with tf.variable_scope('base_pelee', reuse=reuse):
        layers = []
        with tf.variable_scope('steam_block', reuse=reuse):
            output_layer = steam_block(input_tensor_batch, bn=bn, wd=FLAGS.weight_decay)

        with tf.variable_scope('stage_1', reuse=reuse):
            with tf.variable_scope('dense_block'):
                output_layer = dense_block(output_layer, block_config[0], 16, bottleneck_width[0], bn=bn,
                                           wd=FLAGS.weight_decay)
            with tf.variable_scope('transition_layer'):
                output_channel = output_layer.get_shape().as_list()[-1]
                output_layer = transition_layer(output_layer, output_channel, is_pool=True, bn=bn,
                                                wd=FLAGS.weight_decay)

        with tf.variable_scope('stage_2', reuse=reuse):
            with tf.variable_scope('dense_block'):
                output_layer = dense_block(output_layer, block_config[1], 16, bottleneck_width[1], bn=bn,
                                           wd=FLAGS.weight_decay)
            with tf.variable_scope('transition_layer'):
                output_channel = output_layer.get_shape().as_list()[-1]
                output_layer = transition_layer(output_layer, output_channel, is_pool=False, bn=bn,
                                                wd=FLAGS.weight_decay)
            layers.append(output_layer)
            output_layer = tf.nn.max_pool(output_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                          padding='SAME')

        with tf.variable_scope('stage_3', reuse=reuse):
            with tf.variable_scope('dense_block'):
                output_layer = dense_block(output_layer, block_config[2], k, bottleneck_width[2], bn=bn,
                                           wd=FLAGS.weight_decay)
            with tf.variable_scope('transition_layer'):
                # output_channel = output_layer.get_shape().as_list()[-1]
                output_layer = transition_layer(output_layer, 256, is_pool=False, bn=bn,
                                                wd=FLAGS.weight_decay)
            layers.append(output_layer)
            output_layer = tf.nn.max_pool(output_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                          padding='SAME')

        with tf.variable_scope('stage_4', reuse=reuse):
            with tf.variable_scope('dense_block'):
                output_layer = dense_block(output_layer, block_config[3], k, bottleneck_width[3], bn=bn,
                                           wd=FLAGS.weight_decay)
            with tf.variable_scope('transition_layer'):
                # output_channel = output_layer.get_shape().as_list()[-1]
                output_layer = transition_layer(output_layer, 256, is_pool=False, bn=bn,
                                                wd=FLAGS.weight_decay)
            layers.append(output_layer)

        with tf.variable_scope('extension_1', reuse=reuse):
            with tf.variable_scope('dense_block'):
                output_layer = dense_block_ext(output_layer, bn=bn, wd=FLAGS.weight_decay)
            layers.append(output_layer)

        with tf.variable_scope('extension_2', reuse=reuse):
            with tf.variable_scope('dense_block'):
                output_layer = dense_block_ext(output_layer, bn=bn, wd=FLAGS.weight_decay)
            layers.append(output_layer)

    def class_loc_layer(feature_layer):
        input_channel = feature_layer.get_shape().as_list()[-1]
        class_linear_filter = variable_with_weight_decay('class_linear_weights', [1, 1, input_channel, 128],
                                                         wd=FLAGS.weight_decay)
        class_liner_biases = create_variables('class_linear_biases', 128, tf.constant_initializer(0.0))
        class_filter = variable_with_weight_decay('class_weights', [3, 3, 128, ANCHOR_NUM * FLAGS.num_class],
                                                  wd=FLAGS.weight_decay)
        class_biases = create_variables('class_biases', [ANCHOR_NUM * FLAGS.num_class], tf.constant_initializer(-4.595))

        loc_linear_filter = variable_with_weight_decay('loc_linear_weights', [1, 1, input_channel, 128],
                                                       wd=FLAGS.weight_decay)
        loc_liner_biases = create_variables('loc_linear_biases', 128, tf.constant_initializer(0.0))
        loc_filter = variable_with_weight_decay('loc_weights', [3, 3, 128, ANCHOR_NUM * 4],
                                                wd=FLAGS.weight_decay)
        loc_biases = create_variables('loc_biases', [ANCHOR_NUM * 4], tf.constant_initializer(0.0))

        pre_cls_conv = tf.nn.conv2d(feature_layer, class_linear_filter,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
        pre_cls_conv = tf.nn.bias_add(pre_cls_conv, class_liner_biases)
        cls_pred = tf.nn.conv2d(pre_cls_conv, class_filter, strides=[1, 1, 1, 1], padding='SAME')
        cls_pred = tf.nn.bias_add(cls_pred, class_biases)
        # cls_pred = tf.sigmoid(cls_pred)

        pre_loc_conv = tf.nn.conv2d(feature_layer, loc_linear_filter,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
        pre_loc_conv = tf.nn.bias_add(pre_loc_conv, loc_liner_biases)
        loc_pred = tf.nn.conv2d(pre_loc_conv, loc_filter, strides=[1, 1, 1, 1], padding='SAME')
        loc_pred = tf.nn.bias_add(loc_pred, loc_biases)

        # pred_shape = cls_pred.get_shape().as_list()
        cls_pred = tf.reshape(cls_pred, [-1, FLAGS.num_class])
        loc_pred = tf.reshape(loc_pred, [-1, 4])
        return cls_pred, loc_pred

    with tf.variable_scope('feature_pymarid', reuse=reuse):
        with tf.variable_scope('base_feature5'):
            feature5 = layers[4]

        with tf.variable_scope('base_feature4'):
            with tf.variable_scope('conv_layer1'):
                feat_shape = feature5.get_shape().as_list()
                up_feature5 = tf.image.resize_nearest_neighbor(feature5, [feat_shape[1] * 2, feat_shape[2] * 2])
                up_feature5 = conv_bn_relu_layer(up_feature5, [3, 3, 256, 128], stride=1, padding='SAME',
                                                 bn=bn, wd=FLAGS.weight_decay)

            feature4 = tf.concat([layers[3], up_feature5], axis=-1)

        with tf.variable_scope('base_feature3'):
            with tf.variable_scope('conv_layer1'):
                feat_shape = feature4.get_shape().as_list()
                up_feature4 = tf.image.resize_nearest_neighbor(feature4, [feat_shape[1] * 2, feat_shape[2] * 2])
                up_feature4 = conv_bn_relu_layer(up_feature4, [3, 3, feat_shape[-1], 128], stride=1, padding='SAME',
                                                 bn=bn, wd=FLAGS.weight_decay)
            feature3 = tf.concat([layers[2], up_feature4], axis=-1)

        with tf.variable_scope('base_feature2'):
            with tf.variable_scope('conv_layer1'):
                feat_shape = feature3.get_shape().as_list()
                up_feature3 = tf.image.resize_nearest_neighbor(feature3, [feat_shape[1] * 2, feat_shape[2] * 2])
                up_feature3 = conv_bn_relu_layer(up_feature3, [3, 3, feat_shape[-1], 128], stride=1, padding='SAME',
                                                 bn=bn, wd=FLAGS.weight_decay)

            feature2 = tf.concat([layers[1], up_feature3], axis=-1)

            with tf.variable_scope('base_feature1'):
                with tf.variable_scope('conv_layer1'):
                    feat_shape = feature2.get_shape().as_list()
                    up_feature2 = tf.image.resize_nearest_neighbor(feature2, [feat_shape[1] * 2, feat_shape[2] * 2])
                    up_feature2 = conv_bn_relu_layer(up_feature2, [3, 3, feat_shape[-1], 128], stride=1, padding='SAME',
                                                     bn=bn, wd=FLAGS.weight_decay)
                feature1 = tf.concat([layers[0], up_feature2], axis=-1)

    logits = []
    loc = []

    with tf.variable_scope('detection', reuse=reuse):
        with tf.variable_scope('det_layer5'):
            cls_pred5, loc_pred5 = class_loc_layer(feature5)

        with tf.variable_scope('det_layer4'):
            cls_pred4, loc_pred4 = class_loc_layer(feature4)

        with tf.variable_scope('det_layer3'):
            cls_pred3, loc_pred3 = class_loc_layer(feature3)

        with tf.variable_scope('det_layer2'):
            cls_pred2, loc_pred2 = class_loc_layer(feature2)

        with tf.variable_scope('det_layer1'):
            cls_pred1, loc_pred1 = class_loc_layer(feature1)

        logits.append(cls_pred1)
        logits.append(cls_pred2)
        logits.append(cls_pred3)
        logits.append(cls_pred4)
        logits.append(cls_pred5)
        loc.append(loc_pred1)
        loc.append(loc_pred2)
        loc.append(loc_pred3)
        loc.append(loc_pred4)
        loc.append(loc_pred5)
        logits = tf.concat(logits, axis=0)
        loc = tf.concat(loc, axis=0)
    return logits, loc
