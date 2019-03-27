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

tf.app.flags.DEFINE_boolean('is_base_use_ckpt', False,
                            'Whether to load a checkpoint and continue training')

tf.app.flags.DEFINE_boolean('is_ext_use_ckpt', False,
                            'Whether to load a checkpoint and continue training')

tf.app.flags.DEFINE_boolean('quantize_train', False,
                            'Whether to quantize training')


tf.app.flags.DEFINE_integer('num_class', 1, 'num of class')
tf.app.flags.DEFINE_integer('input_num_channel', 3, 'num of class')
tf.app.flags.DEFINE_integer('img_height', 512, 'image height')
tf.app.flags.DEFINE_integer('img_width', 512, 'image width')
tf.app.flags.DEFINE_integer('train_batch_size', 6, 'train batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 1, 'test batch size')
tf.app.flags.DEFINE_integer('train_epoch', 1, 'train epoch')
tf.app.flags.DEFINE_integer('num_gpus', 4, 'number of gpu')
tf.app.flags.DEFINE_integer('num_samples_per_epoch', 150448, 'num_samples_per_epoch')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 2, 'num_epochs_per_decay')
tf.app.flags.DEFINE_integer('training_steps', 800000, 'training_steps')
tf.app.flags.DEFINE_integer('testing_steps', 200, 'testing_steps')
tf.app.flags.DEFINE_integer('select_top_k', 400, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer('keep_top_k', 200, 'Keep top-k detected objects.')

tf.app.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay')
tf.app.flags.DEFINE_float('bn_epsilon', 0.001, 'bn epsilon')
tf.app.flags.DEFINE_float('momentum', 0.9, 'monument for optimizer')
tf.app.flags.DEFINE_float('train_ema_decay', 0.99, 'The decay moving average')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.955, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          'The decay to use for the moving average.'
                          'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_float('nms_thr', 0.45, 'nms threshold')
tf.app.flags.DEFINE_float('conf_thr', 0.01, 'nms threshold')
tf.app.flags.DEFINE_float('overlap_thr', 0.5, 'nms threshold')
tf.app.flags.DEFINE_float('select_threshold', 0.01, 'Selection threshold.')
tf.app.flags.DEFINE_float('nms_threshold', 0.45, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float('matching_threshold', 0.5, 'Matching threshold with groundtruth objects.')


LAYER_BOXES = [6, 6, 6, 6, 6, 6]
BOX_RATIOS = [[1.0],
              [1.0 / 3, 2.0 / 3, 4.0 / 3],
              [1.0 / 3, 2.0 / 3, 4.0 / 3],
              [1.0 / 3, 2.0 / 3, 4.0 / 3],
              [1.0 / 3, 2.0 / 3, 4.0 / 3]]

FIRST_RATIOS = [1.0, 1.0, 2.0, 3.0, 0.5, 1.0/3.0]
BOX_S_MIN = 0.2
FIRST_SCALE = 0.08
NEGPOSRATIO = 3

BLOCK_CONFIG = [3, 4, 8, 8]
BOTTLENECK_WIDTH = [2, 2, 2, 2]

OUT_SHAPES = [[32, 32], [32, 32], [16, 16], [8, 8], [4, 4]]
ANCHOR_SIZE_BOUNDS = [0.10, 0.90]
MEANS = [123., 117., 104.]
SCALE = [3.0 / 64,  3.0 / 32, 3.0 / 16, 3.0 / 8, 3.0 / 4]
NUM_CENTERS = [2, 1, 1, 1, 1]
SCALE_RATIO = [0.02, 0.046875, 0.09375, 0.125, 0.125]
IOU_thr = [0.5, 0.5, 0.5]
NEG_IOU_thr = [0.26, 0.266, 0.287,  0.26, 0.25, 0.36]
Afa = [1.0, 0.7, 0.35, 0.25, 0.175, 0.175]
POW_INDEX = [4, 3.5, 2.5, 2, 1.5, 1.5]
#ANCHOR_NUM = 3
RATIOS_NUM = [1, 3, 3, 3, 3]
SCALE_NUM = 1
WHRATIO = 1.0
