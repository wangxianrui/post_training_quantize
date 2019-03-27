import net
import bboxes_match2
import anchor_ssd
from hyper_parameters import *
import get_loss2
import tf_utils
import time
import os
from gpu_parameters_setting import *
import numpy as np
slim = tf.contrib.slim


def train():
    g = tf.Graph()
    with g.as_default(), tf.device('/cpu:0'):
        anchors = anchor_ssd.anchors_all_layers(OUT_SHAPES, BOX_RATIOS)
        file_name = FLAGS.data_dir + 'merge_train2.tfrecords'
        image, bboxes, labels = tf_utils.decode_tfrecord(file_name)
        glabels, gloc, gscores = bboxes_match2.tf_bboxes_encode(labels,
                                                                bboxes,
                                                                anchors,
                                                                FLAGS.num_class)

        batch_shape = [1] + [1] * 3

        r = tf.train.shuffle_batch(
            tf_utils.reshape_list([image, glabels, gloc, gscores]),
            batch_size=FLAGS.train_batch_size,
            capacity=(800 + FLAGS.num_gpus) * FLAGS.train_batch_size,
            min_after_dequeue=800 * FLAGS.train_batch_size)

        b_image, b_gclasses, b_gloc, b_gscores = tf_utils.reshape_list(r, batch_shape)

        batch_queue = slim.prefetch_queue.prefetch_queue(
            tf_utils.reshape_list([b_image, b_gclasses, b_gloc, b_gscores]),
            capacity=2 * FLAGS.num_gpus)

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                      trainable=False)
        #decay_steps = int(FLAGS.num_samples_per_epoch / FLAGS.train_batch_size / FLAGS.num_gpus *
        #                  FLAGS.num_epochs_per_decay)
        #learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
        #                                            global_step,
        #                                            decay_steps,
        #                                            FLAGS.learning_rate_decay_factor,
        #                                            staircase=True,
        #                                            name='exponential_decay_learning_rate')

        cur_t = tf.cast(global_step, tf.float32)
        learning_rate = 0.00001 + 0.5 * (0.001 - 0.00001) * (1.0 + tf.cos(cur_t / 1000000. * 3.1415926))
        #optimizer = tf.train.RMSPropOptimizer(learning_rate,
        #                                      decay=FLAGS.rmsprop_decay,
        #                                      momentum=FLAGS.rmsprop_momentum,
        #                                      epsilon=FLAGS.opt_epsilon)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

        tower_grads = []
        reuse = False
        if FLAGS.quantize_train:
            num_gpus = 1
        else:
            num_gpus = FLAGS.num_gpus
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                #with tf.name_scope('GPU_%d' % i) as scope:
                b_image, b_gclasses, b_glocalisations, b_gscores = \
                tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)
                #b_image = tf.image.resize_bilinear(b_image, [512, 512]
                b_image = tf.cast(b_image, tf.float32)
                b_image1, b_image2 = tf.split(b_image, 2, axis=0)
                lamda = np.random.beta(1.5, 1.5)
                b_image = lamda * b_image1 + (1. - lamda) * b_image2
                b_image = b_image - tf.constant(MEANS)
                b_image = b_image * 0.017

                logits, localisations = net.inference(b_image, bn=True, reuse=reuse)
                cur_loss = get_loss2.losses(logits, localisations,
                                           b_gclasses, b_glocalisations, b_gscores, lamda,
                                           match_threshold=0.5,
                                           alpha=1.)
                if FLAGS.quantize_train:
                    tf.contrib.quantize.create_training_graph(input_graph=g,
                                                              quant_delay=0)
                #                          scope=scope)
                #var_train_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ssd_extension')
                reuse = True
                grads = optimizer.compute_gradients(cur_loss)
                tower_grads.append(grads)

        grads = get_loss2.average_gradients(tower_grads)


        #variable_averages = tf.train.ExponentialMovingAverage(
        #    FLAGS.moving_average_decay, global_step)
        #variables_averages_op = variable_averages.apply(var_train_list)

        updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updates_op):

            train_op = optimizer.apply_gradients(grads, global_step=global_step)

        sess = tf.Session(config=sess_config)

        base_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='base_pelee')
        if FLAGS.is_base_use_ckpt is True:
            saver = tf.train.Saver(base_vars)
            saver.restore(sess, tf.train.latest_checkpoint('checkpoint/'))
        else:
            sess.run(tf.variables_initializer(base_vars))

        ext_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='detection')
        ext_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature_pymarid')
        ext_vars += ext_vars2

        if FLAGS.is_ext_use_ckpt is True:
            saver = tf.train.Saver(ext_vars)
            saver.restore(sess, tf.train.latest_checkpoint('checkpoint/'))
        else:
            sess.run(tf.variables_initializer(ext_vars))

        g_list = tf.global_variables()
        exclude = [val for val in g_list if val not in base_vars and val not in ext_vars]
        sess.run(tf.variables_initializer(exclude))
        sess.run(tf.local_variables_initializer())

        if not FLAGS.quantize_train:
            var_list = tf.trainable_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            saver = tf.train.Saver(var_list)
        else:
            saver = tf.train.Saver(g_list)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        start_time = time.time()
        for step in range(FLAGS.training_steps):
            _, loss_value = sess.run([train_op, cur_loss])
            if step != 0 and step % 10 == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print(duration)
                format_str = ('step %d, loss = %.2f')
                print(format_str % (step, loss_value))

            if (step % 10000 == 0) or (step + 1) == FLAGS.training_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)

train()
