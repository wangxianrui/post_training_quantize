from hyper_parameters import *
from tensorflow.python.ops import math_ops

# import cv2

slim = tf.contrib.slim


def tf_bboxes_decode_layer(feat_localizations,
                           anchors_layer, ):
    yref, xref, href, wref = anchors_layer

    cx = feat_localizations[:, :, :, :, :, 0] * wref * 0.1 + xref
    cy = feat_localizations[:, :, :, :, :, 1] * href * 0.1 + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, :, 2] * 0.2)
    h = href * tf.exp(feat_localizations[:, :, :, :, :, 3] * 0.2)
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.

    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes


def tf_bboxes_decode(feat_localizations,
                     anchors,
                     scope='bboxes_decode'):
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_bboxes_decode_layer(feat_localizations[i],
                                       anchors_layer))
        return bboxes


def tf_bboxes_select_layer(predictions_layer, localizations_layer,
                           select_threshold=None,
                           num_classes=21,
                           scope=None):
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = get_shape(predictions_layer)
        # p_shape = predictions_layer.get_shape().as_list()
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = get_shape(localizations_layer)
        # l_shape = localizations_layer.get_shape().as_list()
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))

        d_scores = {}
        d_bboxes = {}
        for c in range(num_classes):
            if c < num_classes:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def tf_bboxes_select(predictions_net, localizations_net,
                     select_threshold=None,
                     num_classes=21,
                     scope=None):
    with tf.name_scope(scope, 'bboxes_select',
                       [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = tf_bboxes_select_layer(predictions_net[i],
                                                    localizations_net[i],
                                                    select_threshold,
                                                    num_classes)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
        return d_scores, d_bboxes


def bboxes_sort(scores, bboxes, top_k=400, scope=None):
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_sort_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_sort(scores[c], bboxes[c], top_k=top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):
        # Sort scores...
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        # Trick to be able to use tf.gather: map for each element in the first dim.
        def fn_gather(bboxes, idxes):
            bb = tf.gather(bboxes, idxes)
            return [bb]

        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]),
                      [bboxes, idxes],
                      dtype=[bboxes.dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        bboxes = r[0]
        return scores, bboxes


def get_shape(x, rank=None):
    """Returns the dimensions of a Tensor as list of integers or scale tensors.

    Args:
      x: N-d Tensor;
      rank: Rank of the Tensor. If None, will try to guess it.
    Returns:
      A list of `[d1, d2, ..., dN]` corresponding to the dimensions of the
        input tensor.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def pad_axis(x, offset, size, axis=0, name=None):
    with tf.name_scope(name, 'pad_axis'):
        shape = get_shape(x)
        rank = len(shape)
        # Padding description.
        new_size = tf.maximum(size - offset - shape[axis], 0)
        pad1 = tf.stack([0] * axis + [offset] + [0] * (rank - axis - 1))
        pad2 = tf.stack([0] * axis + [new_size] + [0] * (rank - axis - 1))
        paddings = tf.stack([pad1, pad2], axis=1)
        x = tf.pad(x, paddings, mode='CONSTANT')
        # Reshape, to get fully defined shape if possible.
        # TODO: fix with tf.slice
        shape[axis] = size
        x = tf.reshape(x, tf.stack(shape))
        return x


def bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores,
                                             keep_top_k, nms_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)
        # Pad results.
        scores = pad_axis(scores, 0, keep_top_k, axis=0)
        bboxes = pad_axis(bboxes, 0, keep_top_k, axis=0)
        return scores, bboxes


def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=200,
                     scope=None):
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_nms_batch(scores[c], bboxes[c],
                                        nms_threshold=nms_threshold,
                                        keep_top_k=keep_top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_nms_batch'):
        r = tf.map_fn(lambda x: bboxes_nms(x[0], x[1],
                                           nms_threshold, keep_top_k),
                      (scores, bboxes),
                      dtype=(scores.dtype, bboxes.dtype),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        scores, bboxes = r
        return scores, bboxes


def detected_bboxes(predictions, localisations,
                    select_threshold=None, nms_threshold=0.5,
                    top_k=400, keep_top_k=200):
    # Select top_k bboxes from predictions, and clip
    rscores, rbboxes = \
        tf_bboxes_select(predictions, localisations,
                         select_threshold=select_threshold,
                         num_classes=FLAGS.num_class)
    rscores, rbboxes = \
        bboxes_sort(rscores, rbboxes, top_k=top_k)
    # Apply NMS algorithm.
    rscores, rbboxes = \
        bboxes_nms_batch(rscores, rbboxes,
                         nms_threshold=nms_threshold,
                         keep_top_k=keep_top_k)
    # if clipping_bbox is not None:
    #     rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
    return rscores, rbboxes


def safe_divide(numerator, denominator, name):
    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)


def bboxes_jaccard(bbox_ref, bboxes, name=None):
    with tf.name_scope(name, 'bboxes_jaccard'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[1], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[0], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[1] + bboxes[3], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[0] + bboxes[2], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = -inter_vol \
                    + bboxes[2] * bboxes[3] \
                    + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
        jaccard = safe_divide(inter_vol, union_vol, 'jaccard')
        return jaccard


def bboxes_matching(label, scores, bboxes,
                    glabels, gbboxes,
                    matching_threshold=0.5, scope=None):
    with tf.name_scope(scope, 'bboxes_matching_single',
                       [scores, bboxes, glabels, gbboxes]):
        rsize = tf.size(scores)
        rshape = tf.shape(scores)
        rlabel = tf.cast(label, glabels.dtype)
        # Number of groundtruth boxes.
        n_gbboxes = tf.count_nonzero(tf.equal(glabels, label))
        # Grountruth matching arrays.
        gmatch = tf.zeros(tf.shape(glabels), dtype=tf.bool)
        grange = tf.range(tf.size(glabels), dtype=tf.int32)
        # True/False positive matching TensorArrays.
        sdtype = tf.bool
        ta_tp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)
        ta_fp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)

        # Loop over returned objects.
        def m_condition(i, ta_tp, ta_fp, gmatch):
            r = tf.less(i, rsize)
            return r

        def m_body(i, ta_tp, ta_fp, gmatch):
            # Jaccard score with groundtruth bboxes.
            rbbox = bboxes[i]
            scores_flag = scores[i] > FLAGS.select_threshold
            jaccard = bboxes_jaccard(rbbox, gbboxes)
            jaccard = jaccard * tf.cast(tf.equal(glabels, rlabel), dtype=jaccard.dtype)

            # Best fit, checking it's above threshold.
            idxmax = tf.cast(tf.argmax(jaccard, axis=0), tf.int32)
            jcdmax = jaccard[idxmax]
            match = jcdmax > matching_threshold
            match = tf.logical_and(match, scores_flag)
            existing_match = gmatch[idxmax]

            # TP: match & no previous match and FP: previous match | no match.
            # If difficult: no record, i.e FP=False and TP=False.
            # tp = tf.logical_and(match, tf.logical_not(existing_match))
            tp = match
            ta_tp = ta_tp.write(i, tp)
            fp = tf.logical_or(existing_match, tf.logical_not(match))
            ta_fp = ta_fp.write(i, fp)
            # Update grountruth match.
            mask = tf.logical_and(tf.equal(grange, idxmax), match)
            gmatch = tf.logical_or(gmatch, mask)

            return [i + 1, ta_tp, ta_fp, gmatch]

        # Main loop definition.
        i = 0
        [i, ta_tp_bool, ta_fp_bool, gmatch] = \
            tf.while_loop(m_condition, m_body,
                          [i, ta_tp_bool, ta_fp_bool, gmatch],
                          parallel_iterations=1,
                          back_prop=False)
        # TensorArrays to Tensors and reshape.
        tp_match = tf.reshape(ta_tp_bool.stack(), rshape)
        fp_match = tf.reshape(ta_fp_bool.stack(), rshape)
        # Some debugging information...
        # tp_match = tf.Print(tp_match,
        #                     [n_gbboxes,
        #                      tf.reduce_sum(tf.cast(tp_match, tf.int64)),
        #                      tf.reduce_sum(tf.cast(fp_match, tf.int64)),
        #                      tf.reduce_sum(tf.cast(gmatch, tf.int64))],
        #                     'Matching (NG, TP, FP, GM): ')
        return n_gbboxes, tp_match, fp_match, gmatch


def bboxes_matching_batch(labels, scores, bboxes,
                          glabels, gbboxes,
                          matching_threshold=0.5, scope=None):
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_matching_batch_dict'):
            d_n_gbboxes = {}
            d_tp = {}
            d_fp = {}
            d_gmatch = {}
            for c in labels:
                n, tp, fp, gmatch, _ = bboxes_matching_batch(c, scores[c], bboxes[c],
                                                             glabels, gbboxes,
                                                             matching_threshold)
                d_n_gbboxes[c] = n
                d_tp[c] = tp
                d_fp[c] = fp
                d_gmatch[c] = gmatch
            return d_n_gbboxes, d_tp, d_fp, d_gmatch, scores

    with tf.name_scope(scope, 'bboxes_matching_batch',
                       [scores, bboxes, glabels, gbboxes]):
        r = tf.map_fn(lambda x: bboxes_matching(labels, x[0], x[1],
                                                x[2], x[3],
                                                matching_threshold),
                      (scores, bboxes, glabels, gbboxes),
                      dtype=(tf.int64, tf.bool, tf.bool, tf.bool),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=True,
                      infer_shape=True)
        return r[0], r[1], r[2], r[3], scores


def prediction_fp(logits):
    predictions = []
    for l in logits:
        predictions.append(slim.softmax(l))
    return predictions
