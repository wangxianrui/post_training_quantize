import tensorflow as tf
import numpy as np
from hyper_parameters import *
from tensorflow.python.tools import freeze_graph
import net
import os


def main():
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 512, 512, 3], name='placeholder')

    logits, loc = net.inference(input_placeholder, bn=False, reuse=False)

    saver = tf.train.Saver()
    sess = tf.Session()

    saver.restore(sess, 'ssd_train/model.ckpt-362500')
    saver.save(sess, 'ssd_train/model.ckpt')
    var_train_list = tf.global_variables()
    check_var_list = var_train_list[500:]
    var_v = sess.run(check_var_list)


main()
