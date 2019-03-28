name: "Placeholder"
op: "Placeholder"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: -1
      }
      dim {
        size: 512
      }
      dim {
        size: 512
      }
      dim {
        size: 3
      }
    }
  }
}

name: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000\003\000\000\000\020\000\000\000"
    }
  }
}

name: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.1873171627521515
    }
  }
}

name: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.1873171627521515
    }
  }
}

name: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/weights"
    }
  }
}

name: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/weights"
    }
  }
}

name: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/weights"
    }
  }
}

name: "base_pelee/steam_block/conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv1/weights/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv1/weights"
input: "base_pelee/steam_block/conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv1/weights/read"
op: "Identity"
input: "base_pelee/steam_block/conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/weights"
    }
  }
}

name: "base_pelee/steam_block/conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/steam_block/conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/steam_block/conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/steam_block/conv1/weight_loss"
op: "Mul"
input: "base_pelee/steam_block/conv1/L2Loss"
input: "base_pelee/steam_block/conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/steam_block/conv1/Conv2D"
op: "Conv2D"
input: "Placeholder"
input: "base_pelee/steam_block/conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/steam_block/conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv1/bn/gamma"
input: "base_pelee/steam_block/conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/steam_block/conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/gamma"
    }
  }
}

name: "base_pelee/steam_block/conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/steam_block/conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv1/bn/beta"
input: "base_pelee/steam_block/conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/steam_block/conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/beta"
    }
  }
}

name: "base_pelee/steam_block/conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/steam_block/conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv1/bn/moving_mean"
input: "base_pelee/steam_block/conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/steam_block/conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/steam_block/conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/steam_block/conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv1/bn/moving_variance"
input: "base_pelee/steam_block/conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/steam_block/conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/steam_block/conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/steam_block/conv1/Conv2D"
input: "base_pelee/steam_block/conv1/bn/gamma/read"
input: "base_pelee/steam_block/conv1/bn/beta/read"
input: "base_pelee/steam_block/conv1/bn/moving_mean/read"
input: "base_pelee/steam_block/conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/steam_block/conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/steam_block/conv1/Relu"
op: "Relu"
input: "base_pelee/steam_block/conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\020\000\000\000\020\000\000\000"
    }
  }
}

name: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.4330126941204071
    }
  }
}

name: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.4330126941204071
    }
  }
}

name: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/max"
input: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/weights"
    }
  }
}

name: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/weights"
    }
  }
}

name: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/mul"
input: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/weights"
    }
  }
}

name: "base_pelee/steam_block/conv2_1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 16
      }
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv2_1/weights/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv2_1/weights"
input: "base_pelee/steam_block/conv2_1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_1/weights/read"
op: "Identity"
input: "base_pelee/steam_block/conv2_1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/weights"
    }
  }
}

name: "base_pelee/steam_block/conv2_1/L2Loss"
op: "L2Loss"
input: "base_pelee/steam_block/conv2_1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/steam_block/conv2_1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/steam_block/conv2_1/weight_loss"
op: "Mul"
input: "base_pelee/steam_block/conv2_1/L2Loss"
input: "base_pelee/steam_block/conv2_1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/steam_block/conv2_1/Conv2D"
op: "Conv2D"
input: "base_pelee/steam_block/conv1/Relu"
input: "base_pelee/steam_block/conv2_1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/steam_block/conv2_1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv2_1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv2_1/bn/gamma"
input: "base_pelee/steam_block/conv2_1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_1/bn/gamma/read"
op: "Identity"
input: "base_pelee/steam_block/conv2_1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/gamma"
    }
  }
}

name: "base_pelee/steam_block/conv2_1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/steam_block/conv2_1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv2_1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv2_1/bn/beta"
input: "base_pelee/steam_block/conv2_1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_1/bn/beta/read"
op: "Identity"
input: "base_pelee/steam_block/conv2_1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/beta"
    }
  }
}

name: "base_pelee/steam_block/conv2_1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/steam_block/conv2_1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv2_1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv2_1/bn/moving_mean"
input: "base_pelee/steam_block/conv2_1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/steam_block/conv2_1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/moving_mean"
    }
  }
}

name: "base_pelee/steam_block/conv2_1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/steam_block/conv2_1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv2_1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv2_1/bn/moving_variance"
input: "base_pelee/steam_block/conv2_1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/steam_block/conv2_1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_1/bn/moving_variance"
    }
  }
}

name: "base_pelee/steam_block/conv2_1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/steam_block/conv2_1/Conv2D"
input: "base_pelee/steam_block/conv2_1/bn/gamma/read"
input: "base_pelee/steam_block/conv2_1/bn/beta/read"
input: "base_pelee/steam_block/conv2_1/bn/moving_mean/read"
input: "base_pelee/steam_block/conv2_1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/steam_block/conv2_1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/steam_block/conv2_1/Relu"
op: "Relu"
input: "base_pelee/steam_block/conv2_1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000\020\000\000\000\020\000\000\000"
    }
  }
}

name: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.14433756470680237
    }
  }
}

name: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.14433756470680237
    }
  }
}

name: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/max"
input: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/weights"
    }
  }
}

name: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/weights"
    }
  }
}

name: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/mul"
input: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/weights"
    }
  }
}

name: "base_pelee/steam_block/conv2_2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 16
      }
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv2_2/weights/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv2_2/weights"
input: "base_pelee/steam_block/conv2_2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_2/weights/read"
op: "Identity"
input: "base_pelee/steam_block/conv2_2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/weights"
    }
  }
}

name: "base_pelee/steam_block/conv2_2/L2Loss"
op: "L2Loss"
input: "base_pelee/steam_block/conv2_2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/steam_block/conv2_2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/steam_block/conv2_2/weight_loss"
op: "Mul"
input: "base_pelee/steam_block/conv2_2/L2Loss"
input: "base_pelee/steam_block/conv2_2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/steam_block/conv2_2/Conv2D"
op: "Conv2D"
input: "base_pelee/steam_block/conv2_1/Relu"
input: "base_pelee/steam_block/conv2_2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/steam_block/conv2_2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv2_2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv2_2/bn/gamma"
input: "base_pelee/steam_block/conv2_2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_2/bn/gamma/read"
op: "Identity"
input: "base_pelee/steam_block/conv2_2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/gamma"
    }
  }
}

name: "base_pelee/steam_block/conv2_2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/steam_block/conv2_2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv2_2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv2_2/bn/beta"
input: "base_pelee/steam_block/conv2_2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_2/bn/beta/read"
op: "Identity"
input: "base_pelee/steam_block/conv2_2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/beta"
    }
  }
}

name: "base_pelee/steam_block/conv2_2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/steam_block/conv2_2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv2_2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv2_2/bn/moving_mean"
input: "base_pelee/steam_block/conv2_2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/steam_block/conv2_2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/moving_mean"
    }
  }
}

name: "base_pelee/steam_block/conv2_2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 16
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/steam_block/conv2_2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 16
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv2_2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv2_2/bn/moving_variance"
input: "base_pelee/steam_block/conv2_2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv2_2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/steam_block/conv2_2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv2_2/bn/moving_variance"
    }
  }
}

name: "base_pelee/steam_block/conv2_2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/steam_block/conv2_2/Conv2D"
input: "base_pelee/steam_block/conv2_2/bn/gamma/read"
input: "base_pelee/steam_block/conv2_2/bn/beta/read"
input: "base_pelee/steam_block/conv2_2/bn/moving_mean/read"
input: "base_pelee/steam_block/conv2_2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/steam_block/conv2_2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/steam_block/conv2_2/Relu"
op: "Relu"
input: "base_pelee/steam_block/conv2_2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/steam_block/MaxPool"
op: "MaxPool"
input: "base_pelee/steam_block/conv1/Relu"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "ksize"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}

name: "base_pelee/steam_block/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: -1
    }
  }
}

name: "base_pelee/steam_block/concat"
op: "ConcatV2"
input: "base_pelee/steam_block/conv2_2/Relu"
input: "base_pelee/steam_block/MaxPool"
input: "base_pelee/steam_block/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000 \000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.3061862289905548
    }
  }
}

name: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.3061862289905548
    }
  }
}

name: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/max"
input: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/weights"
    }
  }
}

name: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/weights"
    }
  }
}

name: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/mul"
input: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/weights"
    }
  }
}

name: "base_pelee/steam_block/conv3/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 32
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv3/weights/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv3/weights"
input: "base_pelee/steam_block/conv3/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv3/weights/read"
op: "Identity"
input: "base_pelee/steam_block/conv3/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/weights"
    }
  }
}

name: "base_pelee/steam_block/conv3/L2Loss"
op: "L2Loss"
input: "base_pelee/steam_block/conv3/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/steam_block/conv3/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/steam_block/conv3/weight_loss"
op: "Mul"
input: "base_pelee/steam_block/conv3/L2Loss"
input: "base_pelee/steam_block/conv3/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/steam_block/conv3/Conv2D"
op: "Conv2D"
input: "base_pelee/steam_block/concat"
input: "base_pelee/steam_block/conv3/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv3/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/steam_block/conv3/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv3/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv3/bn/gamma"
input: "base_pelee/steam_block/conv3/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv3/bn/gamma/read"
op: "Identity"
input: "base_pelee/steam_block/conv3/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/gamma"
    }
  }
}

name: "base_pelee/steam_block/conv3/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/steam_block/conv3/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv3/bn/beta/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv3/bn/beta"
input: "base_pelee/steam_block/conv3/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv3/bn/beta/read"
op: "Identity"
input: "base_pelee/steam_block/conv3/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/beta"
    }
  }
}

name: "base_pelee/steam_block/conv3/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/steam_block/conv3/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv3/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv3/bn/moving_mean"
input: "base_pelee/steam_block/conv3/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv3/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/steam_block/conv3/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/moving_mean"
    }
  }
}

name: "base_pelee/steam_block/conv3/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/steam_block/conv3/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/steam_block/conv3/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/steam_block/conv3/bn/moving_variance"
input: "base_pelee/steam_block/conv3/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/steam_block/conv3/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/steam_block/conv3/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/steam_block/conv3/bn/moving_variance"
    }
  }
}

name: "base_pelee/steam_block/conv3/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/steam_block/conv3/Conv2D"
input: "base_pelee/steam_block/conv3/bn/gamma/read"
input: "base_pelee/steam_block/conv3/bn/beta/read"
input: "base_pelee/steam_block/conv3/bn/moving_mean/read"
input: "base_pelee/steam_block/conv3/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/steam_block/conv3/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/steam_block/conv3/Relu"
op: "Relu"
input: "base_pelee/steam_block/conv3/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000 \000\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.25
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.25
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 32
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/steam_block/conv3/Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/gamma/read"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/beta/read"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv1/Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/gamma/read"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/beta/read"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_1/concat"
op: "ConcatV2"
input: "base_pelee/steam_block/conv3/Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_1/left_conv2/Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_1/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000@\000\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.21650634706020355
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.21650634706020355
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 64
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_1/concat"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/gamma/read"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/beta/read"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv1/Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/gamma/read"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/beta/read"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_2/concat"
op: "ConcatV2"
input: "base_pelee/stage_1/dense_block/dense_layer_1/concat"
input: "base_pelee/stage_1/dense_block/dense_layer_2/left_conv2/Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_2/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000`\000\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.19364917278289795
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.19364917278289795
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 96
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_2/concat"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/gamma/read"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/beta/read"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/L2Loss"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv1/Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/gamma/read"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/beta/read"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_1/dense_block/dense_layer_3/concat"
op: "ConcatV2"
input: "base_pelee/stage_1/dense_block/dense_layer_2/concat"
input: "base_pelee/stage_1/dense_block/dense_layer_3/left_conv2/Relu"
input: "base_pelee/stage_1/dense_block/dense_layer_3/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000\200\000\000\000"
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.1530931144952774
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.1530931144952774
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 128
      }
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_1/transition_layer/conv1/weights"
input: "base_pelee/stage_1/transition_layer/conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_1/transition_layer/conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_1/transition_layer/conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_1/transition_layer/conv1/L2Loss"
input: "base_pelee/stage_1/transition_layer/conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_1/dense_block/dense_layer_3/concat"
input: "base_pelee/stage_1/transition_layer/conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_1/transition_layer/conv1/bn/gamma"
input: "base_pelee/stage_1/transition_layer/conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_1/transition_layer/conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_1/transition_layer/conv1/bn/beta"
input: "base_pelee/stage_1/transition_layer/conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_1/transition_layer/conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_1/transition_layer/conv1/bn/moving_mean"
input: "base_pelee/stage_1/transition_layer/conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_1/transition_layer/conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_1/transition_layer/conv1/bn/moving_variance"
input: "base_pelee/stage_1/transition_layer/conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_1/transition_layer/conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_1/transition_layer/conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_1/transition_layer/conv1/Conv2D"
input: "base_pelee/stage_1/transition_layer/conv1/bn/gamma/read"
input: "base_pelee/stage_1/transition_layer/conv1/bn/beta/read"
input: "base_pelee/stage_1/transition_layer/conv1/bn/moving_mean/read"
input: "base_pelee/stage_1/transition_layer/conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_1/transition_layer/conv1/Relu"
op: "Relu"
input: "base_pelee/stage_1/transition_layer/conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_1/transition_layer/MaxPool"
op: "MaxPool"
input: "base_pelee/stage_1/transition_layer/conv1/Relu"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "ksize"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\200\000\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.1767766922712326
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.1767766922712326
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 128
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_1/transition_layer/MaxPool"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/gamma/read"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/beta/read"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv1/Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/gamma/read"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/beta/read"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_1/concat"
op: "ConcatV2"
input: "base_pelee/stage_1/transition_layer/MaxPool"
input: "base_pelee/stage_2/dense_block/dense_layer_1/left_conv2/Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_1/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\240\000\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.16366341710090637
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.16366341710090637
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 160
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_1/concat"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/gamma/read"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/beta/read"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv1/Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/gamma/read"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/beta/read"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_2/concat"
op: "ConcatV2"
input: "base_pelee/stage_2/dense_block/dense_layer_1/concat"
input: "base_pelee/stage_2/dense_block/dense_layer_2/left_conv2/Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_2/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\300\000\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.1530931144952774
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.1530931144952774
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 192
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_2/concat"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/gamma/read"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/beta/read"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv1/Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/gamma/read"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/beta/read"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_3/concat"
op: "ConcatV2"
input: "base_pelee/stage_2/dense_block/dense_layer_2/concat"
input: "base_pelee/stage_2/dense_block/dense_layer_3/left_conv2/Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_3/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\340\000\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.14433756470680237
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.14433756470680237
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 224
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_3/concat"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/gamma/read"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/beta/read"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/L2Loss"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv1/Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/gamma/read"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/beta/read"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_2/dense_block/dense_layer_4/concat"
op: "ConcatV2"
input: "base_pelee/stage_2/dense_block/dense_layer_3/concat"
input: "base_pelee/stage_2/dense_block/dense_layer_4/left_conv2/Relu"
input: "base_pelee/stage_2/dense_block/dense_layer_4/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\000\001\000\000\000\001\000\000"
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.10825317353010178
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.10825317353010178
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 256
      }
      dim {
        size: 256
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_2/transition_layer/conv1/weights"
input: "base_pelee/stage_2/transition_layer/conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_2/transition_layer/conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_2/transition_layer/conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_2/transition_layer/conv1/L2Loss"
input: "base_pelee/stage_2/transition_layer/conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_2/dense_block/dense_layer_4/concat"
input: "base_pelee/stage_2/transition_layer/conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 256
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 256
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_2/transition_layer/conv1/bn/gamma"
input: "base_pelee/stage_2/transition_layer/conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_2/transition_layer/conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 256
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 256
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_2/transition_layer/conv1/bn/beta"
input: "base_pelee/stage_2/transition_layer/conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_2/transition_layer/conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 256
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 256
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_2/transition_layer/conv1/bn/moving_mean"
input: "base_pelee/stage_2/transition_layer/conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_2/transition_layer/conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 256
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 256
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_2/transition_layer/conv1/bn/moving_variance"
input: "base_pelee/stage_2/transition_layer/conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_2/transition_layer/conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_2/transition_layer/conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_2/transition_layer/conv1/Conv2D"
input: "base_pelee/stage_2/transition_layer/conv1/bn/gamma/read"
input: "base_pelee/stage_2/transition_layer/conv1/bn/beta/read"
input: "base_pelee/stage_2/transition_layer/conv1/bn/moving_mean/read"
input: "base_pelee/stage_2/transition_layer/conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_2/transition_layer/conv1/Relu"
op: "Relu"
input: "base_pelee/stage_2/transition_layer/conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_2/MaxPool"
op: "MaxPool"
input: "base_pelee/stage_2/transition_layer/conv1/Relu"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "ksize"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\000\001\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.1369306445121765
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.1369306445121765
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 256
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_2/MaxPool"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv1/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_1/concat"
op: "ConcatV2"
input: "base_pelee/stage_2/MaxPool"
input: "base_pelee/stage_3/dense_block/dense_layer_1/left_conv2/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_1/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000 \001\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.13055823743343353
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.13055823743343353
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 288
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_1/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv1/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_2/concat"
op: "ConcatV2"
input: "base_pelee/stage_3/dense_block/dense_layer_1/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_2/left_conv2/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_2/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000@\001\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.125
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.125
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 320
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_2/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv1/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_3/concat"
op: "ConcatV2"
input: "base_pelee/stage_3/dense_block/dense_layer_2/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_3/left_conv2/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_3/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000`\001\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.1200961172580719
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.1200961172580719
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 352
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_3/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv1/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_4/concat"
op: "ConcatV2"
input: "base_pelee/stage_3/dense_block/dense_layer_3/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_4/left_conv2/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_4/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\200\001\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.1157275140285492
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.1157275140285492
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 384
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_4/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv1/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_5/concat"
op: "ConcatV2"
input: "base_pelee/stage_3/dense_block/dense_layer_4/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_5/left_conv2/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_5/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\240\001\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.11180339753627777
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.11180339753627777
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 416
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_5/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv1/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_6/concat"
op: "ConcatV2"
input: "base_pelee/stage_3/dense_block/dense_layer_5/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_6/left_conv2/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_6/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\300\001\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.10825317353010178
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.10825317353010178
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 448
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_6/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv1/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_7/concat"
op: "ConcatV2"
input: "base_pelee/stage_3/dense_block/dense_layer_6/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_7/left_conv2/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_7/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\340\001\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.10502100735902786
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.10502100735902786
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 480
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_7/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/L2Loss"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv1/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/gamma/read"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/beta/read"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_3/dense_block/dense_layer_8/concat"
op: "ConcatV2"
input: "base_pelee/stage_3/dense_block/dense_layer_7/concat"
input: "base_pelee/stage_3/dense_block/dense_layer_8/left_conv2/Relu"
input: "base_pelee/stage_3/dense_block/dense_layer_8/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\000\002\000\000\000\002\000\000"
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0765465572476387
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0765465572476387
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 512
      }
      dim {
        size: 512
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_3/transition_layer/conv1/weights"
input: "base_pelee/stage_3/transition_layer/conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_3/transition_layer/conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_3/transition_layer/conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_3/transition_layer/conv1/L2Loss"
input: "base_pelee/stage_3/transition_layer/conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/dense_block/dense_layer_8/concat"
input: "base_pelee/stage_3/transition_layer/conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 512
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 512
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_3/transition_layer/conv1/bn/gamma"
input: "base_pelee/stage_3/transition_layer/conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_3/transition_layer/conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 512
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 512
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_3/transition_layer/conv1/bn/beta"
input: "base_pelee/stage_3/transition_layer/conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_3/transition_layer/conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 512
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 512
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_3/transition_layer/conv1/bn/moving_mean"
input: "base_pelee/stage_3/transition_layer/conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_3/transition_layer/conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 512
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 512
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_3/transition_layer/conv1/bn/moving_variance"
input: "base_pelee/stage_3/transition_layer/conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_3/transition_layer/conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_3/transition_layer/conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_3/transition_layer/conv1/Conv2D"
input: "base_pelee/stage_3/transition_layer/conv1/bn/gamma/read"
input: "base_pelee/stage_3/transition_layer/conv1/bn/beta/read"
input: "base_pelee/stage_3/transition_layer/conv1/bn/moving_mean/read"
input: "base_pelee/stage_3/transition_layer/conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_3/transition_layer/conv1/Relu"
op: "Relu"
input: "base_pelee/stage_3/transition_layer/conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_3/MaxPool"
op: "MaxPool"
input: "base_pelee/stage_3/transition_layer/conv1/Relu"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "ksize"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\000\002\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.10206207633018494
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.10206207633018494
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 512
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_3/MaxPool"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv1/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_1/concat"
op: "ConcatV2"
input: "base_pelee/stage_3/MaxPool"
input: "base_pelee/stage_4/dense_block/dense_layer_1/left_conv2/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_1/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000 \002\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.09933992475271225
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.09933992475271225
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 544
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_1/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv1/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_2/concat"
op: "ConcatV2"
input: "base_pelee/stage_4/dense_block/dense_layer_1/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_2/left_conv2/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_2/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000@\002\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.09682458639144897
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.09682458639144897
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 576
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_2/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv1/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_3/concat"
op: "ConcatV2"
input: "base_pelee/stage_4/dense_block/dense_layer_2/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_3/left_conv2/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_3/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000`\002\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.09449111670255661
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.09449111670255661
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 608
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_3/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv1/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_4/concat"
op: "ConcatV2"
input: "base_pelee/stage_4/dense_block/dense_layer_3/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_4/left_conv2/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_4/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\200\002\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.09231861680746078
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.09231861680746078
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 640
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_4/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv1/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_5/concat"
op: "ConcatV2"
input: "base_pelee/stage_4/dense_block/dense_layer_4/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_5/left_conv2/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_5/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\240\002\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0902893915772438
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0902893915772438
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 672
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_5/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv1/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_6/concat"
op: "ConcatV2"
input: "base_pelee/stage_4/dense_block/dense_layer_5/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_6/left_conv2/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_6/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\300\002\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0883883461356163
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0883883461356163
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 704
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_6/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv1/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_7/concat"
op: "ConcatV2"
input: "base_pelee/stage_4/dense_block/dense_layer_6/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_7/left_conv2/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_7/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\340\002\000\000@\000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.08660253882408142
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.08660253882408142
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 736
      }
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_7/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 64
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 64
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000@\000\000\000 \000\000\000"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0833333358168602
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 64
      }
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/L2Loss"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv1/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 32
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 32
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/gamma/read"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/beta/read"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_mean/read"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/Relu"
op: "Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 3
    }
  }
}

name: "base_pelee/stage_4/dense_block/dense_layer_8/concat"
op: "ConcatV2"
input: "base_pelee/stage_4/dense_block/dense_layer_7/concat"
input: "base_pelee/stage_4/dense_block/dense_layer_8/left_conv2/Relu"
input: "base_pelee/stage_4/dense_block/dense_layer_8/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\000\003\000\000\000\002\000\000"
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.06846532225608826
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.06846532225608826
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 768
      }
      dim {
        size: 512
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weights/Assign"
op: "Assign"
input: "base_pelee/stage_4/transition_layer/conv1/weights"
input: "base_pelee/stage_4/transition_layer/conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weights/read"
op: "Identity"
input: "base_pelee/stage_4/transition_layer/conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/weights"
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/stage_4/transition_layer/conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/weight_loss"
op: "Mul"
input: "base_pelee/stage_4/transition_layer/conv1/L2Loss"
input: "base_pelee/stage_4/transition_layer/conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/dense_block/dense_layer_8/concat"
input: "base_pelee/stage_4/transition_layer/conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 512
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 512
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/stage_4/transition_layer/conv1/bn/gamma"
input: "base_pelee/stage_4/transition_layer/conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/stage_4/transition_layer/conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/gamma"
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 512
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 512
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/stage_4/transition_layer/conv1/bn/beta"
input: "base_pelee/stage_4/transition_layer/conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/stage_4/transition_layer/conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/beta"
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 512
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 512
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/stage_4/transition_layer/conv1/bn/moving_mean"
input: "base_pelee/stage_4/transition_layer/conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/stage_4/transition_layer/conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 512
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 512
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/stage_4/transition_layer/conv1/bn/moving_variance"
input: "base_pelee/stage_4/transition_layer/conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/stage_4/transition_layer/conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/stage_4/transition_layer/conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/stage_4/transition_layer/conv1/Conv2D"
input: "base_pelee/stage_4/transition_layer/conv1/bn/gamma/read"
input: "base_pelee/stage_4/transition_layer/conv1/bn/beta/read"
input: "base_pelee/stage_4/transition_layer/conv1/bn/moving_mean/read"
input: "base_pelee/stage_4/transition_layer/conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/stage_4/transition_layer/conv1/Relu"
op: "Relu"
input: "base_pelee/stage_4/transition_layer/conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\000\002\000\000\300\000\000\000"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.09231861680746078
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.09231861680746078
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 512
      }
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/left_conv1/weights"
input: "base_pelee/extension_1/dense_block/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/extension_1/dense_block/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/extension_1/dense_block/left_conv1/L2Loss"
input: "base_pelee/extension_1/dense_block/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/stage_4/transition_layer/conv1/Relu"
input: "base_pelee/extension_1/dense_block/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/gamma"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/beta"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/extension_1/dense_block/left_conv1/Conv2D"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/gamma/read"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/beta/read"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_mean/read"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv1/Relu"
op: "Relu"
input: "base_pelee/extension_1/dense_block/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000\300\000\000\000\300\000\000\000"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0416666679084301
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0416666679084301
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 192
      }
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/left_conv2/weights"
input: "base_pelee/extension_1/dense_block/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/extension_1/dense_block/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/extension_1/dense_block/left_conv2/L2Loss"
input: "base_pelee/extension_1/dense_block/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/extension_1/dense_block/left_conv1/Relu"
input: "base_pelee/extension_1/dense_block/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/gamma"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/beta"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/extension_1/dense_block/left_conv2/Conv2D"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/gamma/read"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/beta/read"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_mean/read"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/extension_1/dense_block/left_conv2/Relu"
op: "Relu"
input: "base_pelee/extension_1/dense_block/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/MaxPool"
op: "MaxPool"
input: "base_pelee/stage_4/transition_layer/conv1/Relu"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "ksize"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\000\002\000\000\300\000\000\000"
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.09231861680746078
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.09231861680746078
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 512
      }
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/right_conv1/weights"
input: "base_pelee/extension_1/dense_block/right_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weights/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/right_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/extension_1/dense_block/right_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/weight_loss"
op: "Mul"
input: "base_pelee/extension_1/dense_block/right_conv1/L2Loss"
input: "base_pelee/extension_1/dense_block/right_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/extension_1/dense_block/right_conv1/MaxPool"
input: "base_pelee/extension_1/dense_block/right_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/gamma"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/beta"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/beta"
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/extension_1/dense_block/right_conv1/Conv2D"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/gamma/read"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/beta/read"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_mean/read"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/extension_1/dense_block/right_conv1/Relu"
op: "Relu"
input: "base_pelee/extension_1/dense_block/right_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_1/dense_block/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: -1
    }
  }
}

name: "base_pelee/extension_1/dense_block/concat"
op: "ConcatV2"
input: "base_pelee/extension_1/dense_block/left_conv2/Relu"
input: "base_pelee/extension_1/dense_block/right_conv1/Relu"
input: "base_pelee/extension_1/dense_block/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\200\001\000\000\300\000\000\000"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.10206207633018494
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.10206207633018494
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 384
      }
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/left_conv1/weights"
input: "base_pelee/extension_2/dense_block/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/extension_2/dense_block/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/extension_2/dense_block/left_conv1/L2Loss"
input: "base_pelee/extension_2/dense_block/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/extension_1/dense_block/concat"
input: "base_pelee/extension_2/dense_block/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/gamma"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/beta"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/extension_2/dense_block/left_conv1/Conv2D"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/gamma/read"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/beta/read"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_mean/read"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv1/Relu"
op: "Relu"
input: "base_pelee/extension_2/dense_block/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000\300\000\000\000\300\000\000\000"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.0416666679084301
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.0416666679084301
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 192
      }
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/left_conv2/weights"
input: "base_pelee/extension_2/dense_block/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/extension_2/dense_block/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/extension_2/dense_block/left_conv2/L2Loss"
input: "base_pelee/extension_2/dense_block/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/extension_2/dense_block/left_conv1/Relu"
input: "base_pelee/extension_2/dense_block/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/gamma"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/beta"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/extension_2/dense_block/left_conv2/Conv2D"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/gamma/read"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/beta/read"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_mean/read"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/extension_2/dense_block/left_conv2/Relu"
op: "Relu"
input: "base_pelee/extension_2/dense_block/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/MaxPool"
op: "MaxPool"
input: "base_pelee/extension_1/dense_block/concat"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "ksize"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\200\001\000\000\300\000\000\000"
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.10206207633018494
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.10206207633018494
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 384
      }
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/right_conv1/weights"
input: "base_pelee/extension_2/dense_block/right_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weights/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/right_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/extension_2/dense_block/right_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/weight_loss"
op: "Mul"
input: "base_pelee/extension_2/dense_block/right_conv1/L2Loss"
input: "base_pelee/extension_2/dense_block/right_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/extension_2/dense_block/right_conv1/MaxPool"
input: "base_pelee/extension_2/dense_block/right_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/gamma"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/beta"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/beta"
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 192
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 192
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/extension_2/dense_block/right_conv1/Conv2D"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/gamma/read"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/beta/read"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_mean/read"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/extension_2/dense_block/right_conv1/Relu"
op: "Relu"
input: "base_pelee/extension_2/dense_block/right_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_2/dense_block/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: -1
    }
  }
}

name: "base_pelee/extension_2/dense_block/concat"
op: "ConcatV2"
input: "base_pelee/extension_2/dense_block/left_conv2/Relu"
input: "base_pelee/extension_2/dense_block/right_conv1/Relu"
input: "base_pelee/extension_2/dense_block/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\200\001\000\000\200\000\000\000"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.10825317353010178
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.10825317353010178
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 384
      }
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/left_conv1/weights"
input: "base_pelee/extension_3/dense_block/left_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weights/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/left_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/extension_3/dense_block/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/weight_loss"
op: "Mul"
input: "base_pelee/extension_3/dense_block/left_conv1/L2Loss"
input: "base_pelee/extension_3/dense_block/left_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/extension_2/dense_block/concat"
input: "base_pelee/extension_3/dense_block/left_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/gamma"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/beta"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/beta"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/extension_3/dense_block/left_conv1/Conv2D"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/gamma/read"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/beta/read"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_mean/read"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv1/Relu"
op: "Relu"
input: "base_pelee/extension_3/dense_block/left_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\003\000\000\000\003\000\000\000\200\000\000\000\200\000\000\000"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.05103103816509247
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.05103103816509247
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/max"
input: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/mul"
input: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 3
      }
      dim {
        size: 3
      }
      dim {
        size: 128
      }
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weights/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/left_conv2/weights"
input: "base_pelee/extension_3/dense_block/left_conv2/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weights/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/left_conv2/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/L2Loss"
op: "L2Loss"
input: "base_pelee/extension_3/dense_block/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/weight_loss"
op: "Mul"
input: "base_pelee/extension_3/dense_block/left_conv2/L2Loss"
input: "base_pelee/extension_3/dense_block/left_conv2/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/Conv2D"
op: "Conv2D"
input: "base_pelee/extension_3/dense_block/left_conv1/Relu"
input: "base_pelee/extension_3/dense_block/left_conv2/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/gamma"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/gamma/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/gamma"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/beta/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/beta"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/beta/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/beta"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance"
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/extension_3/dense_block/left_conv2/Conv2D"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/gamma/read"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/beta/read"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_mean/read"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/extension_3/dense_block/left_conv2/Relu"
op: "Relu"
input: "base_pelee/extension_3/dense_block/left_conv2/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/MaxPool"
op: "MaxPool"
input: "base_pelee/extension_2/dense_block/concat"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "ksize"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 2
      i: 2
      i: 1
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\200\001\000\000\200\000\000\000"
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.10825317353010178
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.10825317353010178
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/max"
input: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/RandomUniform"
input: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform"
op: "Add"
input: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/mul"
input: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 384
      }
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weights/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/right_conv1/weights"
input: "base_pelee/extension_3/dense_block/right_conv1/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weights/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/right_conv1/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/weights"
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/L2Loss"
op: "L2Loss"
input: "base_pelee/extension_3/dense_block/right_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/weight_loss"
op: "Mul"
input: "base_pelee/extension_3/dense_block/right_conv1/L2Loss"
input: "base_pelee/extension_3/dense_block/right_conv1/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/Conv2D"
op: "Conv2D"
input: "base_pelee/extension_3/dense_block/right_conv1/MaxPool"
input: "base_pelee/extension_3/dense_block/right_conv1/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/gamma/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/gamma"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/gamma/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/gamma"
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/beta/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/beta"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/beta/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/beta"
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 0.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean"
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 1.0
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance/Assign"
op: "Assign"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance/read"
op: "Identity"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance"
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "base_pelee/extension_3/dense_block/right_conv1/Conv2D"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/gamma/read"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/beta/read"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_mean/read"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "base_pelee/extension_3/dense_block/right_conv1/Relu"
op: "Relu"
input: "base_pelee/extension_3/dense_block/right_conv1/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "base_pelee/extension_3/dense_block/concat/axis"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: -1
    }
  }
}

name: "base_pelee/extension_3/dense_block/concat"
op: "ConcatV2"
input: "base_pelee/extension_3/dense_block/left_conv2/Relu"
input: "base_pelee/extension_3/dense_block/right_conv1/Relu"
input: "base_pelee/extension_3/dense_block/concat/axis"
attr {
  key: "N"
  value {
    i: 2
  }
}
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
      }
      tensor_content: "\001\000\000\000\001\000\000\000\000\001\000\000\200\000\000\000"
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/min"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: -0.125
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/max"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.125
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/RandomUniform"
op: "RandomUniform"
input: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/shape"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "seed"
  value {
    i: 0
  }
}
attr {
  key: "seed2"
  value {
    i: 0
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/sub"
op: "Sub"
input: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/max"
input: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/weights"
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/mul"
op: "Mul"
input: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/RandomUniform"
input: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/sub"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/weights"
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform"
op: "Add"
input: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/mul"
input: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform/min"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/weights"
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weights"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/weights"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 1
      }
      dim {
        size: 1
      }
      dim {
        size: 256
      }
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weights/Assign"
op: "Assign"
input: "feature_pymarid/base_feature4/conv_layer0/weights"
input: "feature_pymarid/base_feature4/conv_layer0/weights/Initializer/random_uniform"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/weights"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weights/read"
op: "Identity"
input: "feature_pymarid/base_feature4/conv_layer0/weights"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/weights"
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/L2Loss"
op: "L2Loss"
input: "feature_pymarid/base_feature4/conv_layer0/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weight_loss/y"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 9.999999747378752e-05
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/weight_loss"
op: "Mul"
input: "feature_pymarid/base_feature4/conv_layer0/L2Loss"
input: "feature_pymarid/base_feature4/conv_layer0/weight_loss/y"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/Conv2D"
op: "Conv2D"
input: "base_pelee/extension_3/dense_block/concat"
input: "feature_pymarid/base_feature4/conv_layer0/weights/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "dilations"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "padding"
  value {
    s: "SAME"
  }
}
attr {
  key: "strides"
  value {
    list {
      i: 1
      i: 1
      i: 1
      i: 1
    }
  }
}
attr {
  key: "use_cudnn_on_gpu"
  value {
    b: true
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/gamma/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/gamma"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 1.0
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/gamma"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/gamma"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/gamma/Assign"
op: "Assign"
input: "feature_pymarid/base_feature4/conv_layer0/bn/gamma"
input: "feature_pymarid/base_feature4/conv_layer0/bn/gamma/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/gamma"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/gamma/read"
op: "Identity"
input: "feature_pymarid/base_feature4/conv_layer0/bn/gamma"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/gamma"
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/beta/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/beta"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 0.0
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/beta"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/beta"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/beta/Assign"
op: "Assign"
input: "feature_pymarid/base_feature4/conv_layer0/bn/beta"
input: "feature_pymarid/base_feature4/conv_layer0/bn/beta/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/beta"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/beta/read"
op: "Identity"
input: "feature_pymarid/base_feature4/conv_layer0/bn/beta"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/beta"
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/moving_mean/Initializer/zeros"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/moving_mean"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 0.0
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/moving_mean"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/moving_mean"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/moving_mean/Assign"
op: "Assign"
input: "feature_pymarid/base_feature4/conv_layer0/bn/moving_mean"
input: "feature_pymarid/base_feature4/conv_layer0/bn/moving_mean/Initializer/zeros"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/moving_mean"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/moving_mean/read"
op: "Identity"
input: "feature_pymarid/base_feature4/conv_layer0/bn/moving_mean"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/moving_mean"
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/moving_variance/Initializer/ones"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/moving_variance"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 128
        }
      }
      float_val: 1.0
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/moving_variance"
op: "VariableV2"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/moving_variance"
    }
  }
}
attr {
  key: "container"
  value {
    s: ""
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: 128
      }
    }
  }
}
attr {
  key: "shared_name"
  value {
    s: ""
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/moving_variance/Assign"
op: "Assign"
input: "feature_pymarid/base_feature4/conv_layer0/bn/moving_variance"
input: "feature_pymarid/base_feature4/conv_layer0/bn/moving_variance/Initializer/ones"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/moving_variance"
    }
  }
}
attr {
  key: "use_locking"
  value {
    b: true
  }
}
attr {
  key: "validate_shape"
  value {
    b: true
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/moving_variance/read"
op: "Identity"
input: "feature_pymarid/base_feature4/conv_layer0/bn/moving_variance"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer0/bn/moving_variance"
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/FusedBatchNorm"
op: "FusedBatchNorm"
input: "feature_pymarid/base_feature4/conv_layer0/Conv2D"
input: "feature_pymarid/base_feature4/conv_layer0/bn/gamma/read"
input: "feature_pymarid/base_feature4/conv_layer0/bn/beta/read"
input: "feature_pymarid/base_feature4/conv_layer0/bn/moving_mean/read"
input: "feature_pymarid/base_feature4/conv_layer0/bn/moving_variance/read"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.0010000000474974513
  }
}
attr {
  key: "is_training"
  value {
    b: false
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/bn/Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 0.9900000095367432
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/Relu"
op: "Relu"
input: "feature_pymarid/base_feature4/conv_layer0/bn/FusedBatchNorm"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/ResizeNearestNeighbor/size"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 2
        }
      }
      tensor_content: "\010\000\000\000\010\000\000\000"
    }
  }
}

name: "feature_pymarid/base_feature4/conv_layer0/ResizeNearestNeighbor"
op: "ResizeNearestNeighbor"
input: "feature_pymarid/base_feature4/conv_layer0/Relu"
input: "feature_pymarid/base_feature4/conv_layer0/ResizeNearestNeighbor/size"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "align_corners"
  value {
    b: false
  }
}

name: "feature_pymarid/base_feature4/conv_layer1/weights/Initializer/random_uniform/shape"
op: "Const"
attr {
  key: "_class"
  value {
    list {
      s: "loc:@feature_pymarid/base_feature4/conv_layer1/weights"
    }
  }
}
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
        dim {
          size: 4
        }
  }