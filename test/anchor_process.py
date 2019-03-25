import numpy as np
import math
from test.hyper_parameters import *


def anchor_one_layer(scale,
                     scale_ratio,
                     feat_shape,
                     box_ratios):

    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(np.float32) + 0.5) / feat_shape[0]
    x = (x.astype(np.float32) + 0.5) / feat_shape[1] * WHRATIO

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    num_anchors = len(box_ratios) * SCALE_NUM
    h = np.zeros((num_anchors, ), dtype=np.float32)
    w = np.zeros((num_anchors, ), dtype=np.float32)
    temp_scale = scale
    for k in range(SCALE_NUM):
        for i, r in enumerate(box_ratios):
            h[k * RATIOS_NUM + i] = temp_scale / math.sqrt(r)
            w[k * RATIOS_NUM + i] = temp_scale * math.sqrt(r)
        temp_scale = temp_scale * scale_ratio

    return y, x, h, w


def anchors_all_layers(layers_shape,
                       anchor_ratios):
    num_featlayer = len(layers_shape)
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = anchor_one_layer(SCALE[i],
                                         SCALE_RATIO[i],
                                         s,
                                         anchor_ratios)

        layers_anchors.append(anchor_bboxes)
    return layers_anchors



