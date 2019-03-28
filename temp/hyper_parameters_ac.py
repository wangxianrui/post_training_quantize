import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir',
                           '/data/MsCOCO2017/',
                           'data dir')
tf.app.flags.DEFINE_string('train_annotation_file',
                           'annotations/instances_train2017.json',
                           'train annotation file')
tf.app.flags.DEFINE_string('train_img_file',
                           '/train2017',
                           'train img file')
tf.app.flags.DEFINE_string('val_annotation_file',
                           'annotations/instances_val2017.json',
                           'val annotation file ')
tf.app.flags.DEFINE_string('val_img_file',
                           'val2017',
                           'val img file')
tf.app.flags.DEFINE_string('test_annotation_file',
                           'image_info_test2017/annotations/image_info_test2017.json',
                           'test annotation file')
tf.app.flags.DEFINE_string('test_img_file',
                           'test2017/test2017',
                           'test img file')

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'checkpoint dir')

tf.app.flags.DEFINE_string('ckpt_path',
                           'E:/object detection tensorflow/object detection/checkpoint/',
                           'checkpoint path')
tf.app.flags.DEFINE_string('test_ckpt_path',
                           'E:/object detection tensorflow/object detection/checkpoint/',
                           'test checkpoint path')

tf.app.flags.DEFINE_boolean('is_base_use_ckpt', True,
                            'Whether to load a checkpoint and continue training')

tf.app.flags.DEFINE_boolean('is_ext_use_ckpt', True,
                            'Whether to load a checkpoint and continue training')


tf.app.flags.DEFINE_integer('num_class', 1, 'num of class')
tf.app.flags.DEFINE_integer('input_num_channel', 3, 'num of class')
tf.app.flags.DEFINE_integer('img_height', 320, 'image height')
tf.app.flags.DEFINE_integer('img_width', 320, 'image width')
tf.app.flags.DEFINE_integer('train_batch_size', 16, 'train batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 1, 'test batch size')
tf.app.flags.DEFINE_integer('train_epoch', 1, 'train epoch')
tf.app.flags.DEFINE_integer('num_gpus', 4, 'number of gpu')
tf.app.flags.DEFINE_integer('num_samples_per_epoch', 108962, 'num_samples_per_epoch')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 2, 'num_epochs_per_decay')
tf.app.flags.DEFINE_integer('training_steps', 500000, 'training_steps')
tf.app.flags.DEFINE_integer('testing_steps', 4544, 'testing_steps')
tf.app.flags.DEFINE_integer('select_top_k', 400, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer('keep_top_k', 200, 'Keep top-k detected objects.')

tf.app.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay')
tf.app.flags.DEFINE_float('bn_epsilon', 0.001, 'bn epsilon')
tf.app.flags.DEFINE_float('momentum', 0.9, 'monument for optimizer')
tf.app.flags.DEFINE_float('train_ema_decay', 0.99, 'The decay moving average')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          'The decay to use for the moving average.'
                          'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_float('nms_thr', 0.45, 'nms threshold')
tf.app.flags.DEFINE_float('conf_thr', 0.01, 'nms threshold')
tf.app.flags.DEFINE_float('overlap_thr', 0.5, 'nms threshold')
tf.app.flags.DEFINE_float('select_threshold', 0.9, 'Selection threshold.')
tf.app.flags.DEFINE_float('nms_threshold', 0.45, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float('matching_threshold', 0.5, 'Matching threshold with groundtruth objects.')


LAYER_BOXES = [6, 6, 6, 6, 6, 6]
BOX_RATIOS = [0.375, 0.75, 1.5]
FIRST_RATIOS = [1.0, 1.0, 2.0, 3.0, 0.5, 1.0/3.0]
BOX_S_MIN = 0.2
FIRST_SCALE = 0.08
NEGPOSRATIO = 3

BLOCK_CONFIG = [6, 8, 8, 6]
BOTTLENECK_WIDTH = [2, 2, 2, 2]

OUT_SHAPES = [[80, 80], [40, 40], [20, 20], [10, 10], [5, 5]]
ANCHOR_SIZE_BOUNDS = [0.10, 0.90]
MEANS = [127.5, 127.5, 127.5]
SCALE = [0.0375, 0.1, 0.225, 0.425, 0.625]
SCALE_RATIO = [0.025, 0.05, 0.1, 0.1, 0.2]
IOU_thr = [0.5, 0.5, 0.5]
NEG_IOU_thr = [0.26, 0.266, 0.287,  0.26, 0.25, 0.36]
Afa = [1.0, 0.7, 0.35, 0.25, 0.175, 0.175]
POW_INDEX = [4, 3.5, 2.5, 2, 1.5, 1.5]
ANCHOR_NUM = 6
RATIOS_NUM = 3
SCALE_NUM = 2
