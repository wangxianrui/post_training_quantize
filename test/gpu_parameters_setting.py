import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
gpu_options.allow_growth = True
sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
