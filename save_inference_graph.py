import tensorflow as tf
import numpy as np
from hyper_parameters import *
from tensorflow.python.tools import freeze_graph
import net
import os


def main():
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 512, 512, 3])

    logits, loc = net.inference(input_placeholder, bn=False, reuse=False)
    for log in logits:
        print(log.shape)
    for l in loc:
        print(l.shape)
    exit()


    g = tf.get_default_graph()
    # tf.contrib.quantize.create_eval_graph(input_graph=g)

    with open('ssd_eval_full/model.pbtxt', 'w') as f:
        f.write(str(g.as_graph_def()))
    saver = tf.train.Saver()
    sess = tf.Session()

    model_path = 'ssd_eval_full/model.ckpt'
    saver.restore(sess, model_path)
    saver.save(sess, 'ssd_eval_full/model.ckpt')
    var_train_list = tf.global_variables()
    check_var_list = var_train_list[500:]
    var_v = sess.run(check_var_list)

    # freeze_graph.freeze_graph(input_graph='quantize_ac/model.pbtxt',
    #                           input_saver='',
    #                           input_binary=False,
    #                           input_checkpoint=model_path,
    #                           output_node_names="detection/concat,detection/concat_1",
    #                           restore_op_name='save/restore_all',
    #                           filename_tensor_name='save/Const:0',
    #                           output_graph='quantize_ac/model.pb',
    #                           clear_devices=True,
    #                           initializer_nodes="")
    # output_node_names="detection/concat,detection/concat_1"


main()
