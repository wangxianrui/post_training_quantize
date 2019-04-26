import tensorflow as tf
from tensorflow.lite.python import lite_constants
import numpy as np
import os


class Config:
    model_dir = 'ssd_eval_quant'
    pb_path = os.path.join(model_dir, 'model.pb')
    input_name = 'pb_input'
    output_name = [
        # logits
        'detection/logits',
        # locations
        'detection/loc'
    ]
    mean_values = 127.500001
    std_values = 127.5


def export_tflite_file():
    converter = tf.lite.TFLiteConverter.from_frozen_graph(Config.pb_path, [Config.input_name], Config.output_name)
    # quantize parameters
    converter.post_training_quantize = True
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {Config.input_name: {Config.mean_values, Config.std_values}}
    tflite_model = converter.convert()
    with tf.gfile.GFile(os.path.join(Config.model_dir, 'model_quant.tflite'), 'wb') as file:
        file.write(tflite_model)


def test_tflite():
    # restore the model and allocate tensors
    interpreter = tf.lite.Interpreter(os.path.join(Config.model_dir, 'model_quant.tflite'))
    interpreter.allocate_tensors()

    # get input & output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('input details: {}'.format(input_details))
    print('output details: {}'.format(output_details))

    net_input_data = np.random.randint(0, 255, [1, 512, 512, 3]).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], net_input_data)
    interpreter.invoke()
    net_output_data = interpreter.get_tensor(output_details[0]['index'])
    std_, mean_ = output_details[0]['quantization']
    print((net_output_data - mean_) * std_)


def main():
    export_tflite_file()
    test_tflite()


if __name__ == '__main__':
    main()
