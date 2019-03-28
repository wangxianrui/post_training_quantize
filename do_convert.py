import tensorflow as tf
from tensorflow.lite.python import lite_constants
import os


class Config:
    model_dir = 'quantize_ac'
    pb_path = os.path.join(model_dir, 'model.pb')
    input_name = 'placeholder'
    output_name = ["detection/concat", "detection/concat_1"]
    mean_values = 128
    std_values = 127


def export_tflite_file():
    converter = tf.lite.TFLiteConverter.from_frozen_graph(Config.pb_path, [Config.input_name], Config.output_name)
    # quantize parameters
    converter.post_training_quantize = True
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {Config.input_name: {Config.mean_values, Config.std_values}}
    tflite_model = converter.convert()
    with tf.gfile.GFile(os.path.join(os.path.join(Config.model_dir, 'model_quan.tflite')), 'wb') as file:
        file.write(tflite_model)


def main():
    export_tflite_file()


if __name__ == '__main__':
    main()
