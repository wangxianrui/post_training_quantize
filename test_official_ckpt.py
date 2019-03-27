import tensorflow as tf
import os
import json
from tqdm import trange
import cv2


def get_img(img_path):
    with tf.Graph().as_default():
        sess = tf.Session()

        with tf.gfile.GFile(img_path, 'rb') as file:
            encoded_jpeg = file.read()
        images = tf.image.decode_jpeg(encoded_jpeg, 3)
        images = tf.expand_dims(images, 0)
        images = tf.image.resize_images(images, (320, 320))
        # images = tf.cast(images, tf.float32) - tf.constant([127.5, 127.5, 127.5])
        # images = images * 0.017
        images = tf.image.per_image_standardization(images)
        return sess.run(images)


def build_graph():
    with tf.Graph().as_default():
        sess = tf.Session()
        # graph
        ckpt_path = 'ssdlite_mobilenet_v2/model.ckpt'
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        net_input = sess.graph.get_tensor_by_name('image_tensor:0')
        net_output = []
        net_output.append(sess.graph.get_tensor_by_name('detection_boxes:0'))
        net_output.append(sess.graph.get_tensor_by_name('detection_scores:0'))
        net_output.append(sess.graph.get_tensor_by_name('detection_classes:0'))
        net_output.append(sess.graph.get_tensor_by_name('num_detections:0'))
    return sess, net_input, net_output


def dump_eval(img_path, detection, json_result):
    img = cv2.imread(img_path)
    height, width, channel = img.shape
    image_id = int(os.path.splitext(os.path.split(img_path)[-1])[0])
    nb_res = int(detection[-1][0])
    for i in range(nb_res):
        bbox = list(detection[0][0][i])
        bbox[0], bbox[1], bbox[2], bbox[3] = bbox[1], bbox[0], bbox[3], bbox[2]
        bbox[2] = (bbox[2] - bbox[0]) * width
        bbox[3] = (bbox[3] - bbox[1]) * height
        bbox[0] = bbox[0] * width
        bbox[1] = bbox[1] * height
        score = float(detection[1][0][i])
        cla = int(detection[2][0][i])
        pred = {
            'image_id': image_id,
            'category_id': cla,
            'bbox': bbox,
            'score': score,
        }
        json_result.append(pred)


def main(_):
    img_dir = '/home/wxrui/dataset/mscoco_2017/images/val2017'
    img_list = [os.path.join(img_dir, name) for name in os.listdir(img_dir) if 'jpg' in name]
    # sess, net_input, net_output = build_graph()

    json_result = []
    # for i in trange(len(img_list)):
    for i in trange(10):
        img_path = img_list[i]
        img = get_img(img_path)
        print(img)
        exit()
        # detection = sess.run(net_output, feed_dict={net_input: img})
        # dump_eval(img_path, detection, json_result)
    json.dump(json_result, open('load_ckpt_coco/result.json', 'w'), indent=4)


if __name__ == '__main__':
    tf.app.run()
