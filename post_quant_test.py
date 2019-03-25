import tensorflow as tf
import numpy as np


def model(inputs, is_train):
    outputs = inputs
    outputs = tf.layers.SeparableConv2D(8, 3, 1, 'same', use_bias=False)(outputs)
    outputs = tf.layers.BatchNormalization(
        beta_initializer=tf.random_uniform_initializer(),
        gamma_initializer=tf.random_uniform_initializer(),
        moving_mean_initializer=tf.random_uniform_initializer(),
        moving_variance_initializer=tf.random_uniform_initializer(),
    )(outputs, is_train)
    outputs = tf.nn.relu6(outputs)
    outputs = tf.layers.SeparableConv2D(8, 3, 1, 'same', use_bias=False)(outputs)
    outputs = tf.layers.BatchNormalization(
        beta_initializer=tf.random_uniform_initializer(),
        gamma_initializer=tf.random_uniform_initializer(),
        moving_mean_initializer=tf.random_uniform_initializer(),
        moving_variance_initializer=tf.random_uniform_initializer(),
    )(outputs, is_train)
    outputs = tf.nn.relu6(outputs)
    return outputs


def get_batch_images():
    for i in range(4):
        np.random.seed(i)
        yield np.random.random([32, 300, 300, 3])


def create_model(save_path):
    with tf.Graph().as_default():
        sess = tf.Session()
        tf.set_random_seed(-1)
        x = tf.placeholder(tf.float32, [None, 300, 300, 3])
        y = model(x, False)
        init = tf.global_variables_initializer()
        tf.add_to_collection('images_final', x)
        tf.add_to_collection('logits_final', y)
        sess.run(init)
        saver = tf.train.Saver()
        saver.save(sess, save_path)


def quantize_test(full_ckpt_path, quant_ckpt_path):
    for batch_images in get_batch_images():
        with tf.Graph().as_default():
            sess = tf.Session()
            saver = tf.train.import_meta_graph(full_ckpt_path + '.meta')
            saver.restore(sess, full_ckpt_path)
            net_input = tf.get_collection('images_final')[0]
            net_output = tf.get_collection('logits_final')[0]
            print('original')
            print(sess.run(net_output, feed_dict={net_input: batch_images}))

        with tf.Graph().as_default():
            sess = tf.Session()
            saver = tf.train.import_meta_graph(quant_ckpt_path + '.meta')
            saver.restore(sess, quant_ckpt_path)
            net_input = tf.get_collection('images_final')[0]
            net_output = tf.get_collection('logits_final')[0]
            print('quantized')
            print(sess.run(net_output, feed_dict={net_input: batch_images}))


def main(_):
    # prepare ckpt file
    full_ckpt_path = 'full_model/model.ckpt'
    quant_ckpt_path = 'quant_model/model.ckpt'
    create_model(full_ckpt_path)

    with tf.Graph().as_default():
        sess = tf.Session()
        saver = tf.train.import_meta_graph(full_ckpt_path + '.meta')
        saver.restore(sess, full_ckpt_path)
        tf.graph_util.remove_training_nodes(sess.graph_def)
        tf.contrib.quantize.create_eval_graph()
        net_input = tf.get_collection('images_final')[0]
        net_output = tf.get_collection('logits_final')[0]

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

        # save
        saver = tf.train.Saver()
        saver.save(sess, quant_ckpt_path)

    quantize_test(full_ckpt_path, quant_ckpt_path)


if __name__ == '__main__':
    tf.app.run()
