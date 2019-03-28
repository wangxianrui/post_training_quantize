import tensorflow as tf
import os
import numpy as np


class Config:
    model_dir = 'quantize_ac'
    ckpt_path = os.path.join(model_dir, 'model.ckpt')
    input_name = 'Placeholder'
    input_shape = [None, 320, 320, 3]
    output_name = ["detection/concat", "detection/concat_1"]


def export_pb_file():
    with tf.Graph().as_default():
        sess = tf.Session()
        net_input = tf.placeholder(tf.float32, Config.input_shape, 'placeholder')
        saver = tf.train.import_meta_graph(Config.ckpt_path + '.meta', clear_devices=True, input_map={
            Config.input_name: net_input})
        saver.restore(sess, Config.ckpt_path)
        net_output = [sess.graph.get_tensor_by_name(name + ':0') for name in Config.output_name]

        graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                 [tsr.op.name for tsr in net_output])
        with tf.gfile.GFile(os.path.join(Config.model_dir, 'model.pb'), 'wb') as file:
            file.write(graph_def.SerializeToString())


def test_pb():
    with tf.Graph().as_default():
        sess = tf.Session()
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(os.path.join(Config.model_dir, 'model.pb'), 'rb') as file:
            graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def)

        net_input = sess.graph.get_tensor_by_name('import/placeholder:0')
        input_shape = [1] + Config.input_shape[1:]
        input_data = np.random.random(input_shape)
        net_output = [sess.graph.get_tensor_by_name('import/' + name + ':0') for name in Config.output_name]
        res = sess.run(net_output, feed_dict={net_input: input_data})
        for r in res:
            print(r.shape)


def main():
    export_pb_file()
    test_pb()


if __name__ == '__main__':
    main()
