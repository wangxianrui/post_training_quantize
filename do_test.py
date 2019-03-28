from gpu_parameters_setting import sess_config
import post_process
from hyper_parameters import *
import numpy as np


class Config:
    res_tfrds = 'val_dense512.tfrecords'
    num_samples = 200


def decode_tfrecord(filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/object/gbbox/gxmin': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/gbbox/gymin': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/gbbox/gwidth': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/gbbox/gheight': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/gbbox/glabel': tf.VarLenFeature(dtype=tf.int64),
                                           'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/score': tf.VarLenFeature(dtype=tf.float32),
                                       })

    glabels = tf.sparse_tensor_to_dense(features['image/object/gbbox/glabel'])
    gxmin = tf.sparse_tensor_to_dense(features['image/object/gbbox/gxmin'])
    gymin = tf.sparse_tensor_to_dense(features['image/object/gbbox/gymin'])
    gw = tf.sparse_tensor_to_dense(features['image/object/gbbox/gwidth'])
    gh = tf.sparse_tensor_to_dense(features['image/object/gbbox/gheight'])
    xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
    ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'])
    ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'])
    scores = tf.sparse_tensor_to_dense(features['image/object/bbox/score'])
    gxmin = tf.reshape(gxmin, [-1, 1])
    gymin = tf.reshape(gymin, [-1, 1])
    gw = tf.reshape(gw, [-1, 1])
    gh = tf.reshape(gh, [-1, 1])
    xmin = tf.reshape(xmin, [-1, 1])
    ymin = tf.reshape(ymin, [-1, 1])
    xmax = tf.reshape(xmax, [-1, 1])
    ymax = tf.reshape(ymax, [-1, 1])
    gbboxes = tf.concat(values=[gxmin, gymin, gw, gh], axis=1)
    bboxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=1)
    return gbboxes, bboxes, glabels, scores


def test(filename, score_threshold, num_pic):
    gbboxes, bboxes, glabels, scores = decode_tfrecord(filename)
    rscores = {}
    rbboxes = {}
    rbboxes[0] = tf.expand_dims(bboxes, axis=0)
    rscores[0] = tf.expand_dims(scores, axis=0)
    b_glabels = tf.expand_dims(glabels, axis=0)
    b_gbboxes = tf.expand_dims(gbboxes, axis=0)
    num_gbboxes, tp, fp, gmatch, rscores = \
        post_process.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                           b_glabels, b_gbboxes,
                                           matching_threshold=score_threshold)
    pre_flag = tf.greater(rscores[0][0], FLAGS.select_threshold)
    fpre_flag = tf.cast(pre_flag, tf.float32)
    pre_num = tf.cast(tf.reduce_sum(fpre_flag), tf.int32)
    tp_flag = tf.logical_and(pre_flag, tp[0][0])
    ftp_flag = tf.cast(tp_flag, tf.float32)
    pos_num = tf.cast(tf.reduce_sum(ftp_flag), tf.int32)

    imatch_flag = tf.cast(gmatch[0][0], tf.int64)
    match_num = tf.reduce_sum(imatch_flag)

    sess = tf.Session(config=sess_config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    total_pre_num = 0
    total_pos_num = 0
    total_gbboxes_num = 0
    total_match_num = 0

    for step in range(num_pic):
        pre_num_v, pos_num_v, num_gbboxes_v, match_num_v, tp_v, fp_v = sess.run(
            [pre_num,
             pos_num,
             num_gbboxes,
             match_num,
             tp,
             fp])

        # gbboxes_v, bboxes_v, glabels_v, scores_v = sess.run([gbboxes, bboxes, glabels, scores])
        # print(step)

        total_pre_num += pre_num_v
        total_pos_num += pos_num_v
        # temp_presion = total_pos_num * 100 / total_pre_num

        total_gbboxes_num += num_gbboxes_v[0][0]
        total_match_num += match_num_v
        # temp_recall = total_match_num * 100 / total_gbboxes_num
        # print(temp_presion)
        # print(temp_recall)

    recall = total_match_num * 100 / total_gbboxes_num
    precision = total_pos_num * 100 / total_pre_num
    print(total_match_num)
    print(total_gbboxes_num)
    print(total_pos_num)
    print(total_pre_num)

    return precision, recall


def map(filename):
    recall = []
    precision = []
    score_threshold = 0.5
    num_pic = Config.num_samples
    ap = np.zeros(shape=[11])
    thr = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.95, 0.96,
           0.98, 0.99, 0.999]
    for i in range(len(thr)):

        print(i)
        FLAGS.select_threshold = thr[i]

        temp_presion, temp_recall = test(filename, score_threshold, num_pic)
        print(temp_presion)
        print(temp_recall)
        if i == 0 or (i > 0 and temp_recall != recall[-1]):
            precision.append(temp_presion)
            recall.append(temp_recall)
    precision = np.array(precision)
    recall = np.array(recall)
    for j in range(10):
        cur_recall = j * 10
        dis_recall = recall - cur_recall
        if dis_recall[0] < 0:
            index0 = 0
            index1 = 1
        else:
            for n in range(len(dis_recall) - 1):
                if dis_recall[n] >= 0 and dis_recall[n + 1] < 0:
                    break
            index0 = n
            index1 = n + 1
        recall1 = recall[index0] - cur_recall
        recall2 = recall[index1] - cur_recall
        drecall = recall2 - recall1
        precision1 = precision[index0]
        precision2 = precision[index1]
        cur_pre = recall2 * precision1 / drecall - recall1 * precision2 / drecall
        if cur_pre < 0:
            cur_pre = 0
        if cur_pre > 100:
            cur_pre = 100
        ap[j] = cur_pre
    print(ap)
    print(np.sum(ap) - ap[0] * 0.5)


filename = Config.res_tfrds
map(filename)
