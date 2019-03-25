import tensorflow as tf
import numpy as np
import os
from random import shuffle


class config:
    num_samples = 128
    batch_size = 32
    img_size = (300, 300)
    img_dir = '/home/wxrui/dataset/mscoco_2017/images/val2017'


def get_batch_images():
    sess = tf.Session()
    imgs_list = [os.path.join(config.img_dir, name) for name in os.listdir(config.img_dir)]
    shuffle(imgs_list)
    for i in range(int(config.num_samples / config.batch_size)):
        batch_images = []
        img_list = imgs_list[i * config.batch_size:(i + 1) * config.batch_size]
        for img_path in img_list:
            with tf.gfile.GFile(img_path, 'rb') as file:
                encoded_jpeg = file.read()
            images = tf.image.decode_jpeg(encoded_jpeg, 3)
            images = tf.expand_dims(images, 0)
            images = tf.image.resize_images(images, config.img_size)
            batch_images.append(images)
        batch_images = tf.concat(batch_images, 0)
        yield sess.run(batch_images)


def main(_):
    # prepare ckpt file
    full_ckpt_path = 'full_model/model.ckpt'
    quant_ckpt_path = 'quant_model/model.ckpt'

    with tf.Graph().as_default():
        sess = tf.Session()
        saver = tf.train.import_meta_graph(full_ckpt_path + '.meta')
        saver.restore(sess, full_ckpt_path)
        tf.graph_util.remove_training_nodes(sess.graph_def)
        tf.contrib.quantize.create_eval_graph()
        net_input = sess.graph.get_tensor_by_name('image_tensor:0')

        # quantize weight
        print('weight quantize ... ')
        for op in sess.graph.get_operations():
            if op.name.endswith('weights_quant/FakeQuantWithMinMaxVars'):
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
                print('{:50} have been quantized'.format(weight_name))

        # quantize activation
        print('activation quantize ... ')
        for op in sess.graph.get_operations():
            if op.name.endswith('act_quant/FakeQuantWithMinMaxVars'):
                act_name = op.inputs[0].name
                min_name = op.inputs[1].name.replace('/read', '')
                max_name = op.inputs[2].name.replace('/read', '')
                min_val = []
                max_val = []
                for batch_images in get_batch_images():
                    act_val = sess.run(act_name, feed_dict={net_input: batch_images})
                    min_val.append(np.min(act_val))
                    max_val.append(np.max(act_val))
                sess.run([
                    tf.assign(sess.graph.get_tensor_by_name(min_name), np.mean(min_val)),
                    tf.assign(sess.graph.get_tensor_by_name(max_name), np.mean(max_val)),
                ])
                print('{:50} have been quantized'.format(act_name))

        saver = tf.train.Saver()
        saver.save(sess, quant_ckpt_path)


if __name__ == '__main__':
    tf.app.run()
