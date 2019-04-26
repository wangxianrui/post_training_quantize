import re

import numpy as np
import tensorflow as tf
from scipy import stats

import tf_utils

slim = tf.contrib.slim

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', default=1, help='')
tf.flags.DEFINE_integer('num_samples', default=1, help='')


class Config:
    num_samples = FLAGS.num_samples
    batch_size = FLAGS.batch_size
    ori_tfrds = 'coco_test.tfrecords'
    full_ckpt_path = 'ssd_eval_full/model.ckpt'
    quant_ckpt_path = 'ssd_eval_quant/model.ckpt'
    input_name = 'ckpt_input'
    input_shape = [None, 512, 512, 3]


def get_batch_images():
    MEANS = [127.5, 127.5, 127.5]
    sess = tf.Session()
    val_path = Config.ori_tfrds
    image0, bboxes, labels = tf_utils.decode_tfrecord(val_path)
    image = tf.train.batch(tf_utils.reshape_list([image0]), batch_size=Config.batch_size)
    batch_queue = slim.prefetch_queue.prefetch_queue(tf_utils.reshape_list([image]))

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    batch_images = []
    for i in range(int(Config.num_samples / Config.batch_size)):
        b_image = batch_queue.dequeue()
        b_image = tf.cast(b_image, tf.float32) - tf.constant(MEANS)
        b_image = b_image / 127.5
        batch_images.append(sess.run(b_image))
    return batch_images


def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


def get_minmax(data, method):
    if method == 0:
        return np.min(data), np.max(data)
    distribution, bins = np.histogram(data, 2048, (np.min(data), np.max(data)))
    distribution = distribution[1:]
    target_bins = 256
    kl_divergence = []  # divergence, i, j
    left, right = 0, len(distribution) - 1
    while (right - left) > target_bins:
        # change left and right towards medium
        if distribution[left] < distribution[right]:
            left += 1
            while distribution[left] == 0:
                left += 1
        else:
            right -= 1
            while distribution[right] == 0:
                right -= 1
        length = right - left + 1

        # distribution p
        p = np.zeros(length)
        p[0] = np.sum(distribution[:left + 1])
        p[1:-1] = distribution[left + 1:right]
        p[-1] = np.sum(distribution[right:])
        non_zeros = (p != 0).astype(np.int)

        # distrubution quantized
        quantized = np.zeros(target_bins)
        num_merges = length // target_bins
        for i in range(target_bins):
            start = i * num_merges
            end = start + num_merges
            quantized[i] = np.sum(p[start:end])
        quantized[-1] += np.sum(p[num_merges * target_bins:])

        # distribution q
        q = np.zeros(length)
        for i in range(target_bins):
            start = i * num_merges
            if i == target_bins - 1:
                end = -1
            else:
                end = start + num_merges
            norm = np.sum(non_zeros[start:end])
            if norm:
                q[start:end] = quantized[i] / norm
        if np.sum(q):
            p = _smooth_distribution(p)
            q = _smooth_distribution(q)
            kl_divergence.append([stats.entropy(p, q), left, right])
    min_diver, left, right = sorted(kl_divergence)[0]
    return bins[left], bins[right]


def main():
    batch_images = get_batch_images()
    with tf.Graph().as_default():
        sess = tf.Session()
        saver = tf.train.import_meta_graph(Config.full_ckpt_path + '.meta', clear_devices=True)
        saver.restore(sess, Config.full_ckpt_path)
        tf.graph_util.remove_training_nodes(sess.graph_def)
        tf.contrib.quantize.create_eval_graph()

        # classify operations
        weight_pattern = r'weights_quant(_\d)?/FakeQuantWithMinMaxVars'
        weight_quant_ops = []
        act_quant_ops = []
        for op in sess.graph.get_operations():
            if op.name.endswith('/FakeQuantWithMinMaxVars'):
                if re.search(weight_pattern, op.name):
                    weight_quant_ops.append(op)
                else:
                    act_quant_ops.append(op)

        # quantize weight
        for op in weight_quant_ops:
            weight_name = op.inputs[0].name
            min_name = op.inputs[1].name.replace('/read', '')
            max_name = op.inputs[2].name.replace('/read', '')
            weight_val = sess.run(weight_name)
            min_val, max_val = get_minmax(weight_val, method=0)
            sess.run([
                tf.assign(sess.graph.get_tensor_by_name(min_name), min_val),
                tf.assign(sess.graph.get_tensor_by_name(max_name), max_val),
            ])
            print('{:100} weights quantize_ac'.format(weight_name))

        # quantize activation
        net_input = sess.graph.get_tensor_by_name(Config.input_name + ':0')
        for op in act_quant_ops:
            act_name = op.inputs[0].name
            min_name = op.inputs[1].name.replace('/read', '')
            max_name = op.inputs[2].name.replace('/read', '')
            min_val = []
            max_val = []
            for images in batch_images:
                act_val = sess.run(act_name, feed_dict={net_input: images})
                minmax_val = get_minmax(act_val, method=0)
                min_val.append(minmax_val[0])
                max_val.append(minmax_val[1])
            sess.run([
                tf.assign(sess.graph.get_tensor_by_name(min_name), np.mean(min_val)),
                tf.assign(sess.graph.get_tensor_by_name(max_name), np.mean(max_val)),
            ])
            print('{:100} activation quantize_ac'.format(act_name))

        # save
        saver = tf.train.Saver()
        saver.save(sess, Config.quant_ckpt_path)


if __name__ == '__main__':
    main()
