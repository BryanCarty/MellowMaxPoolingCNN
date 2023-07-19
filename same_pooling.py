import numpy as np
import tensorflow as tf
import math
import sys


class CustomMaxPooling2D(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomMaxPooling2D, self).__init__()
        self.padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        self.transpose_format = tf.constant([0, 3, 1, 2])
        self.reverse_transpose_format = tf.constant([0, 2, 3, 1])
    def build(self, input_shape):
        _, self.height, self.width, self.channels = input_shape
        self.output_width = tf.floor((self.height-1)/2)+1
    def call(self, inputs):
        self.output= tf.zeros(shape=(tf.shape(inputs)[0], self.channels, self.output_width, self.output_width))
        if self.width%self.output_width!=0:
            inputs = tf.pad(tensor=inputs, paddings=self.padding_format, mode='constant')
        inputs = tf.transpose(inputs, perm=self.transpose_format)
        for b in range(tf.shape(inputs)[0]):
            for c in range(self.channels):
                for w in range(self.output_width):
                    for h in range(self.output_width):
                        self.output[b,c,w,h] = tf.maximum(input_data[w*2:(w+1)*2, h*2:(h+1)*2])
        return tf.transpose(self.output, perm=self.reverse_transpose_format)





def same_max_pooling(input_data):
    output_shape = math.floor((input_data.shape[0] - 1) / 2) + 1
    output_arr = np.zeros(shape=(output_shape,output_shape))
    add_padding = input_data.shape[0]%output_shape!=0
    if add_padding:
        input_data = tf.pad(tensor=input_data, paddings=[[0, 1], [0, 1]], mode='constant')
    for i in range(output_shape):
        for j in range(output_shape):
            output_arr[i, j] = np.max(input_data[i*2:(i+1)*2, j*2:(j+1)*2])
    return output_arr




input_data = np.array([[1, 2, 3, 4, 5],
                       [5, 6, 7, 8, 9],
                       [9, 10, 11, 12, 10],
                       [13, 14, 15, 16, 17],
                       [18, 19, 20, 21, 21]])
'''
input_data = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]])
'''
pool_size = (2, 2)

output = same_max_pooling(input_data)

print(output)