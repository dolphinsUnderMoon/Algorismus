import numpy as np
import tensorflow as tf


class Config:
    batch_size = 32


caps_net_config = Config()


class CapsLayer:
    def __init__(self, num_capsules, capsule_vector_length, with_routing=True, layer_type="FC"):
        self.num_capsules = num_capsules
        self.capsule_vector_length = capsule_vector_length
        self.with_routing = with_routing
        self.layer_type = layer_type

        self.kernel_size = None
        self.stride = None

    def __call__(self, input, kernel_size, stride):
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                assert input.get_shape() == [caps_net_config.batch_size, 20, 20, 256]

                capsules = []
                for i in range(self.capsule_vector_length):
                    with tf.variable_scope('ConvUnit_' + str(i)):
                        capsule = tf.contrib.layers.conv2d(input,
                                                           self.num_capsules,
                                                           self.kernel_size,
                                                           self.stride,
                                                           padding='VALID',
                                                           activation_fn=None)
                        capsule = tf.reshape(capsule, shape=[caps_net_config.batch_size, -1, 1, 1])
                        capsules.append(capsule)

                assert capsules[0].get_shape() == [caps_net_config.batch_size, 1152, 1, 1]
                capsules = tf.concat(capsules, axis=2)
