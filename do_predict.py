import tensorflow as tf
from hyper_parameters import *
import anchor_ssd
import tf_utils
import post_process
from tqdm import trange
import os


class Config:
    model_dir = 'ssd_eval_full'
    ckpt_path = os.path.join(model_dir, 'model.ckpt')
    res_tfrds = 'val_dense512.tfrecords'
    ori_tfrds = 'coco_test.tfrecords'
    test_samples = 200
    input_name = ['Placeholder']
    output_name = [
        # logits
        [
            'detection/det_layer1/Sigmoid',
            'detection/det_layer2/Sigmoid',
            'detection/det_layer3/Sigmoid',
            'detection/det_layer4/Sigmoid',
            'detection/det_layer5/Sigmoid',
        ],
        # locations
        [
            'detection/det_layer1/BiasAdd_3',
            'detection/det_layer2/BiasAdd_3',
            'detection/det_layer3/BiasAdd_3',
            'detection/det_layer4/BiasAdd_3',
            'detection/det_layer5/BiasAdd_3',
        ]
    ]


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


def main():
    writer = tf.python_io.TFRecordWriter(Config.res_tfrds)

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # network
        sess = tf.Session()
        saver = tf.train.import_meta_graph(Config.ckpt_path + '.meta')
        saver.restore(sess, Config.ckpt_path)
        net_input = sess.graph.get_tensor_by_name(Config.input_name[0] + ':0')
        logits = [sess.graph.get_tensor_by_name(name + ':0') for name in Config.output_name[0]]
        localisations = [sess.graph.get_tensor_by_name(name + ':0') for name in Config.output_name[1]]

        logits[0] = tf.reshape(logits[0], [1, 32, 32, 4, 1, 1])
        logits[1] = tf.reshape(logits[1], [1, 32, 32, 1, 3, 1])
        logits[2] = tf.reshape(logits[2], [1, 16, 16, 1, 3, 1])
        logits[3] = tf.reshape(logits[3], [1, 8, 8, 1, 3, 1])
        logits[4] = tf.reshape(logits[4], [1, 4, 4, 1, 3, 1])

        localisations[0] = tf.reshape(localisations[0], [1, 32, 32, 4, 1, 4])
        localisations[1] = tf.reshape(localisations[1], [1, 32, 32, 1, 3, 4])
        localisations[2] = tf.reshape(localisations[2], [1, 16, 16, 1, 3, 4])
        localisations[3] = tf.reshape(localisations[3], [1, 8, 8, 1, 3, 4])
        localisations[4] = tf.reshape(localisations[4], [1, 4, 4, 1, 3, 4])

        anchors = anchor_ssd.anchors_all_layers(OUT_SHAPES, BOX_RATIOS)
        localisations = post_process.tf_bboxes_decode(localisations, anchors)
        rscores, rbboxes = post_process.detected_bboxes(logits, localisations,
                                                        select_threshold=FLAGS.select_threshold,
                                                        nms_threshold=FLAGS.nms_threshold,
                                                        top_k=FLAGS.select_top_k,
                                                        keep_top_k=FLAGS.keep_top_k)
        # data
        data_sess = tf.Session()
        MEANS = [123., 117., 104.]
        val_path = Config.ori_tfrds
        image0, bboxes, labels = tf_utils.decode_tfrecord(val_path)
        b_image = tf.expand_dims(image0, axis=0)
        # b_image = tf.image.resize_bilinear(b_image, [320, 320])
        b_image = tf.cast(b_image, tf.float32) - tf.constant(MEANS)
        b_image = b_image * 0.017
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=data_sess, coord=coord)

    for step in trange(Config.test_samples):
        image_v, b_gbboxes_v, b_glabels_v = data_sess.run([b_image, bboxes, labels])
        rscores_v, rbboxes_v = sess.run([rscores, rbboxes], feed_dict={net_input: image_v})

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


if __name__ == '__main__':
    main()
