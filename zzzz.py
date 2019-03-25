import tensorflow as tf
import numpy as np

sm_writer = tf.summary.FileWriter('log')

with tf.Graph().as_default():
    sess = tf.Session()
    full_ckpt_path = 'test/ssd_eval_full/model.ckpt'
    saver = tf.train.import_meta_graph(full_ckpt_path + '.meta', clear_devices=True)
    saver.restore(sess, full_ckpt_path)
    tf.graph_util.remove_training_nodes(sess.graph_def)
    tf.contrib.quantize.create_eval_graph()

    net_input = sess.graph.get_tensor_by_name('Placeholder:0')
    net_output = [
        sess.graph.get_tensor_by_name('detection/det_layer5/Reshape:0'),
        sess.graph.get_tensor_by_name('detection/det_layer5/Reshape_1:0'),
        sess.graph.get_tensor_by_name('detection/det_layer4/Reshape:0'),
        sess.graph.get_tensor_by_name('detection/det_layer4/Reshape_1:0'),
        sess.graph.get_tensor_by_name('detection/det_layer3/Reshape:0'),
        sess.graph.get_tensor_by_name('detection/det_layer3/Reshape_1:0'),
        sess.graph.get_tensor_by_name('detection/det_layer2/Reshape:0'),
        sess.graph.get_tensor_by_name('detection/det_layer2/Reshape_1:0'),
        sess.graph.get_tensor_by_name('detection/det_layer1/Reshape:0'),
        sess.graph.get_tensor_by_name('detection/det_layer1/Reshape_1:0'),
    ]

    import re

    weight_pattern = r'weights_quant(_\d)?/FakeQuantWithMinMaxVars'
    weight_quant_op = []
    act_quant_op = []
    res = []
    for op in sess.graph.get_operations():
        if op.name.endswith('/FakeQuantWithMinMaxVars'):
            if re.search(weight_pattern, op.name):
                weight_quant_op.append(op)
            else:
                act_quant_op.append(op)
    for op in weight_quant_op:
        print(op.name)
    print('****************')
    for op in act_quant_op:
        print(op.name)
    #
    # rr = []
    # for op in sess.graph.get_operations():
    #     if op.name.endswith('FakeQuantWithMinMaxVars'):
    #         rr.append(op)
    # print(len(rr))
    #
    # # for op in set(rr) - set(res):
    # #     print(op)

sm_writer.add_graph(sess.graph)
sm_writer.close()
