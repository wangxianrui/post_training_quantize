import tensorflow as tf
import numpy as np
import re
import tf_utils

slim = tf.contrib.slim


class config:
    num_samples = 128
    batch_size = 16
    ori_tfrds = 'coco_test.tfrecords'
    full_ckpt_path = 'ssd_eval_full/model.ckpt'
    quant_ckpt_path = 'ssd_eval_quant/model.ckpt'
    input_name = 'Placeholder'


def get_batch_images():
    MEANS = [123., 117., 104.]
    sess = tf.Session()
    val_path = config.ori_tfrds
    image0, bboxes, labels = tf_utils.decode_tfrecord(val_path)
    image = tf.train.batch(tf_utils.reshape_list([image0]), batch_size=config.batch_size)
    batch_queue = slim.prefetch_queue.prefetch_queue(tf_utils.reshape_list([image]))

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    batch_images = []
    for i in range(int(config.num_samples / config.batch_size)):
        b_image = batch_queue.dequeue()
        b_image = tf.cast(b_image, tf.float32) - tf.constant(MEANS)
        b_image = b_image * 0.017
        batch_images.append(sess.run(b_image))
    return batch_images


def main():
    batch_images = get_batch_images()
    with tf.Graph().as_default():
        sess = tf.Session()
        saver = tf.train.import_meta_graph(config.full_ckpt_path + '.meta')
        saver.restore(sess, config.full_ckpt_path)
        tf.graph_util.remove_training_nodes(sess.graph_def)
        tf.contrib.quantize.create_eval_graph()
        net_input = sess.graph.get_tensor_by_name(':0')

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
            min_val = np.min(weight_val)
            max_val = np.max(weight_val)
            sess.run([
                tf.assign(sess.graph.get_tensor_by_name(min_name), min_val),
                tf.assign(sess.graph.get_tensor_by_name(max_name), max_val),
            ])
            print('{:100} weights quantize_ac'.format(weight_name))

        # quantize activation
        for op in act_quant_ops:
            act_name = op.inputs[0].name
            min_name = op.inputs[1].name.replace('/read', '')
            max_name = op.inputs[2].name.replace('/read', '')
            min_val = []
            max_val = []
            for images in batch_images:
                act_val = sess.run(act_name, feed_dict={net_input: images})
                min_val.append(np.min(act_val))
                max_val.append(np.max(act_val))
            sess.run([
                tf.assign(sess.graph.get_tensor_by_name(min_name), np.mean(min_val)),
                tf.assign(sess.graph.get_tensor_by_name(max_name), np.mean(max_val)),
            ])
            print('{:100} activation quantize_ac'.format(act_name))

        # save
        saver = tf.train.Saver()
        saver.save(sess, config.quant_ckpt_path)


if __name__ == '__main__':
    main()
