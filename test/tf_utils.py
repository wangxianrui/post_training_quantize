# import cv2
from test.hyper_parameters import *


def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.

    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i + s])
            i += s
    return r


def decode_tfrecord(filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/width': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/height': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
                                       })

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [FLAGS.img_height, FLAGS.img_width, 3])
    labels = tf.sparse_tensor_to_dense(features['image/object/bbox/label'])
    xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
    ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    w = tf.sparse_tensor_to_dense(features['image/object/bbox/width'])
    h = tf.sparse_tensor_to_dense(features['image/object/bbox/height'])
    # labels = tf.cast(labels, tf.float32)
    xmin = tf.reshape(xmin, [-1, 1])
    ymin = tf.reshape(ymin, [-1, 1])
    w = tf.reshape(w, [-1, 1])
    h = tf.reshape(h, [-1, 1])
    bboxes = tf.concat(values=[xmin, ymin, w, h], axis=1)

    return image, bboxes, labels


def decode_tfrecord3(filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/image_name': tf.FixedLenFeature([], tf.string),
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/width': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/height': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
                                       })

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image_name = features['image/image_name']
    image = tf.reshape(image, [FLAGS.img_height, FLAGS.img_width, 3])
    labels = tf.sparse_tensor_to_dense(features['image/object/bbox/label'])
    xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
    ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    w = tf.sparse_tensor_to_dense(features['image/object/bbox/width'])
    h = tf.sparse_tensor_to_dense(features['image/object/bbox/height'])
    # labels = tf.cast(labels, tf.float32)
    xmin = tf.reshape(xmin, [-1, 1])
    ymin = tf.reshape(ymin, [-1, 1])
    w = tf.reshape(w, [-1, 1])
    h = tf.reshape(h, [-1, 1])
    bboxes = tf.concat(values=[xmin, ymin, w, h], axis=1)

    return image, image_name, bboxes, labels


def decode_tfrecord2(filename, img_height, img_width):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/width': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/height': tf.VarLenFeature(dtype=tf.float32),
                                           'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
                                       })

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [img_height, img_width, 3])
    labels = tf.sparse_tensor_to_dense(features['image/object/bbox/label'])
    xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
    ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    w = tf.sparse_tensor_to_dense(features['image/object/bbox/width'])
    h = tf.sparse_tensor_to_dense(features['image/object/bbox/height'])
    # labels = tf.cast(labels, tf.float32)
    xmin = tf.reshape(xmin, [-1, 1])
    ymin = tf.reshape(ymin, [-1, 1])
    w = tf.reshape(w, [-1, 1])
    h = tf.reshape(h, [-1, 1])
    bboxes = tf.concat(values=[xmin, ymin, w, h], axis=1)

    return image, bboxes, labels