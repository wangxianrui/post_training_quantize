import tensorflow as tf
import os

path = 'ssd_eval_quant/model.ckpt'
with tf.Graph().as_default():
    sess = tf.Session()
    saver = tf.train.import_meta_graph(path + '.meta', clear_devices=True)
    saver.restore(sess, path)
    sm_writer = tf.summary.FileWriter('log', sess.graph)
    sm_writer.close()

os.system('tensorboard --logdir log --host 127.0.0.1')
