import tensorflow as tf

sm_writer = tf.summary.FileWriter('log')
with tf.Graph().as_default():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('ssd_eval_full/model.ckpt' + '.meta')
    saver.restore(sess, 'ssd_eval_full/model.ckpt')

    sm_writer.add_graph(sess.graph)
sm_writer.close()
