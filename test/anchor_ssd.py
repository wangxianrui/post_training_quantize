import numpy as np
import math
from test.hyper_parameters import *


def anchor_one_layer(scale,
                     n,
                     feat_shape,
                     box_ratios):
    stride = 1. / n
    x_layer = []
    y_layer = []
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    for i in range(n):
        for j in range(n):
            y_temp = (y.astype(np.float32) + stride * (i + 0.5)) / feat_shape[0]
            y_layer.append(y_temp)

    for i in range(n):
        for j in range(n):
            x_temp = (x.astype(np.float32) + stride * (j + 0.5)) / feat_shape[1]
            x_layer.append(x_temp)

    y_layer = np.stack(y_layer, axis=-1)
    x_layer = np.stack(x_layer, axis=-1)
    y_layer = np.expand_dims(y_layer, axis=-1)
    x_layer = np.expand_dims(x_layer, axis=-1)
    ratios_num = len(box_ratios)
    h = np.zeros((ratios_num, ), dtype=np.float32)
    w = np.zeros((ratios_num, ), dtype=np.float32)

    for i, r in enumerate(box_ratios):
        h[i] = scale / math.sqrt(r)
        w[i] = scale * math.sqrt(r)

    return y_layer, x_layer, h, w


def anchors_all_layers(layers_shape,
                       anchor_ratios):
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = anchor_one_layer(SCALE[i],
                                         NUM_CENTERS[i],
                                         s,
                                         anchor_ratios[i])

        layers_anchors.append(anchor_bboxes)
    return layers_anchors

'''
anchors = anchors_all_layers(OUT_SHAPES, BOX_RATIOS)
print(len(anchors))
for i, anchors_layer in enumerate(anchors):
    yref_layer, xref_layer, href_layer, wref_layer = anchors_layer
    ymin_layer = yref_layer - href_layer / 2.
    xmin_layer = xref_layer - wref_layer / 2.
    ymax_layer = yref_layer + href_layer / 2.
    xmax_layer = xref_layer + wref_layer / 2.
'''






