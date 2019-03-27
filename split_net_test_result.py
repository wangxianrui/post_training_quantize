from hyper_parameters import *
import anchor_ssd
import tf_utils
import post_process
import net
from gpu_parameters_setting import *

# import cv2
slim = tf.contrib.slim
from tqdm import trange


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def generate_batch(image, image_size, patch_size, step_h, stride, step_w):
    image_batch = []
    anchor_bias = []
    for i in range(step_h):
        start_h = i * stride
        end_h = start_h + patch_size
        bias_y = start_h / image_size
        for j in range(step_w):
            start_w = j * stride
            end_w = start_w + patch_size
            img = image[start_h: end_h, start_w: end_w, :]
            image_batch.append(img)
            bias_x = start_w / image_size
            anchor_bias.append([[[[bias_y, bias_x, bias_y, bias_x]]]])
    image_batch = tf.convert_to_tensor(image_batch)
    anchor_bias = tf.convert_to_tensor(anchor_bias)
    return image_batch, anchor_bias


def build_graph_small(image_batch, anchor_bias, anchors, scale, reuse):
    b_image = tf.cast(image_batch, tf.float32) - tf.constant(MEANS)
    b_image = b_image * 0.017
    logits, localisations = net.inference(b_image, bn=False, reuse=reuse)
    localisations = post_process.tf_bboxes_decode(localisations, anchors)

    '''**********************************************'''
    for i in range(len(logits)):
        localisations[i] = tf.add(scale * localisations[i], anchor_bias)
        localisations[i] = tf.reshape(localisations[i], [1, -1, 4])
        logits[i] = tf.reshape(logits[i], [1, -1, 1])
    return logits, localisations


def test():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        anchors = anchor_ssd.anchors_all_layers(OUT_SHAPES, BOX_RATIOS)
        val_path = 'coco_test.tfrecords'
        image0, bboxes, labels = tf_utils.decode_tfrecord(val_path)

        all_logits = []
        all_localisations = []

        with tf.device('/gpu:%d' % 3):

            b_image = tf.expand_dims(image0, axis=0)
            #b_image = tf.image.resize_bilinear(b_image, [320, 320])
            b_image = tf.cast(b_image, tf.float32) - tf.constant(MEANS)
            b_image = b_image * 0.017
            logits, localisations = net.inference(b_image, bn=False, reuse=False)
            localisations = post_process.tf_bboxes_decode(localisations, anchors)
            for i in range(len(logits)):
                all_logits.append(logits[i])
                all_localisations.append(localisations[i])

        with tf.device('/gpu:%d' % 3):

            rscores, rbboxes = \
                post_process.detected_bboxes(all_logits, all_localisations,
                                             select_threshold=FLAGS.select_threshold,
                                             nms_threshold=FLAGS.nms_threshold,
                                             top_k=FLAGS.select_top_k,
                                             keep_top_k=FLAGS.keep_top_k)

        sess = tf.Session(config=sess_config)

        middle_var1 = tf.global_variables(scope='base_pelee')
        middle_var2 = tf.global_variables(scope='feature_pymarid')
        middle_var3 = tf.global_variables(scope='detection')
        # middle_var3 = tf.global_variables(scope='ssd_extension')
        middle_var = middle_var1 + middle_var2 + middle_var3
        saver = tf.train.Saver(middle_var)
        model_path = 'ssd_eval_full/model.ckpt-512'
        saver.restore(sess, model_path)

        '''***************************************************'''
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        writer = tf.python_io.TFRecordWriter('val_dense512_full.tfrecords')

        for step in trange(FLAGS.testing_steps):
            # print(step)

            # image_v, b_glabels_v, b_gbboxes_v = sess.run([image0, labels, bboxes])
            # continue

            image_v, b_glabels_v, b_gbboxes_v, rscores_v, rbboxes_v = \
                sess.run([image0, labels, bboxes, rscores, rbboxes])

            i = 0
            anns = []
            scores = []

            while i < 200 and rscores_v[0][0][i] > 0:
                rec = rbboxes_v[0][0][i]
                anns.append(rec)
                scores.append(rscores_v[0][0][i])
                # post_process.draw_ann(image_v, rec, 'p{}'.format(i))
                i = i + 1
            xmin = []
            ymin = []
            xmax = []
            ymax = []

            for b in anns:
                assert len(b) == 4
                # pylint: disable=expression-not-assigned
                [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/object/gbbox/gxmin': _float_feature(b_gbboxes_v[:, 0].tolist()),
                'image/object/gbbox/gymin': _float_feature(b_gbboxes_v[:, 1].tolist()),
                'image/object/gbbox/gwidth': _float_feature(b_gbboxes_v[:, 2].tolist()),
                'image/object/gbbox/gheight': _float_feature(b_gbboxes_v[:, 3].tolist()),
                'image/object/gbbox/glabel': _int64_feature(b_glabels_v.tolist()),
                'image/object/bbox/ymin': _float_feature(ymin),
                'image/object/bbox/xmin': _float_feature(xmin),
                'image/object/bbox/ymax': _float_feature(ymax),
                'image/object/bbox/xmax': _float_feature(xmax),
                'image/object/bbox/score': _float_feature(scores),
            }))

            writer.write(example.SerializeToString())

            # Img = cv2.cvtColor(image_v, cv2.COLOR_RGB2BGR)
            # cv2.imshow("patch", Img)
            # cv2.waitKey(0)

        coord.request_stop()
        coord.join(threads)
        writer.close()


test()
