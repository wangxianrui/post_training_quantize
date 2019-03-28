#!/usr/bin/env bash
toco \
    --input_file="model.pb" \
    --output_file="model.tflite" \
    --input_format="TENSORFLOW_GRAPHDEF" \
    --output_format="TFLITE" \
    --inference_type="QUANTIZED_UINT8" \
    --input_arrays="Placeholder" \
    --output_arrays="detection/det_layer1/Reshape,detection/det_layer2/Reshape,detection/det_layer3/Reshape,detection/det_layer4/Reshape,detection/det_layer5/Reshape,detection/det_layer1/Reshape_1,detection/det_layer2/Reshape_1,detection/det_layer3/Reshape_1,detection/det_layer4/Reshape_1,detection/det_layer5/Reshape_1" \
    --input_shapes="1,512,512,3" \
    --mean_values="128" \
    --std_values="127"



toco \
    --input_file="model.pb" \
    --output_file="model.tflite" \
    --input_format="TENSORFLOW_GRAPHDEF" \
    --output_format="TFLITE" \
    --inference_type="QUANTIZED_UINT8" \
    --input_arrays="Placeholder" \
    --output_arrays="detection/concat,detection/concat_1" \
    --input_shapes="1,320,320,3" \
    --mean_values="128" \
    --std_values="127"



toco \
    --graph_def_file="model.pb" \
    --output_file="model.tflite" \
    --input_format="TENSORFLOW_GRAPHDEF" \
    --output_format="TFLITE" \
    --inference_type="QUANTIZED_UINT8" \
    --input_arrays="Placeholder" \
    --output_arrays="detection/concat,detection/concat_1" \
    --input_shapes="1,320,320,3" \
    --mean_values="128" \
    --std_dev_values="127"