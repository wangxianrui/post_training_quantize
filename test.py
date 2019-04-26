import tensorflow as tf
import numpy as np
import os


class Config:
    model_dir = 'ssd_eval_quant'
    ckpt_path = os.path.join(model_dir, 'model.ckpt')
    input_name = 'Placeholder'
    input_shape = [None, 512, 512, 3]
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
            'detection/det_layer1/act_quant_3/FakeQuantWithMinMaxVars',
            'detection/det_layer2/act_quant_3/FakeQuantWithMinMaxVars',
            'detection/det_layer3/act_quant_3/FakeQuantWithMinMaxVars',
            'detection/det_layer4/act_quant_3/FakeQuantWithMinMaxVars',
            'detection/det_layer5/act_quant_3/FakeQuantWithMinMaxVars',
        ]
    ]


with tf.Graph().as_default():
    sess = tf.Session()
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(os.path.join(Config.model_dir, 'model.pb'), 'rb') as file:
        graph_def.ParseFromString(file.read())
    tf.import_graph_def(graph_def)
    # saver = tf.train.import_meta_graph(os.path.join(Config.model_dir, 'model.ckpt.meta'))
    # saver.restore(sess, os.path.join(Config.model_dir, 'model.ckpt'))

    writer = tf.summary.FileWriter('log')
    writer.add_graph(sess.graph)
    writer.close()

    # for op in sess.graph.get_operations():
    #     print(op.name)
    #     input()
    # exit()
    #
    # net_input = sess.graph.get_tensor_by_name('import/placeholder:0')
    # input_shape = [1] + Config.input_shape[1:]
    # input_data = np.random.random(input_shape)
    # net_output = [sess.graph.get_tensor_by_name('import/' + name + ':0') for name in Config.output_name[0]]
    # res = sess.run(net_output, feed_dict={net_input: input_data})
    # for r in res:
    #     print(r.shape)
