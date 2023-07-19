import tensorflow as tf
import sys


def get_mellow_max_fn(w):
    @tf.function
    def mellow_max(x):
        c_val = tf.reduce_max(x)
        numerator = tf.math.log(tf.multiply(0.25, tf.reduce_sum(tf.math.exp([tf.multiply(w, x[0]-c_val), tf.multiply(w, x[1]-c_val), tf.multiply(w, x[2]-c_val), tf.multiply(w, x[3]-c_val)]))))
        return tf.add(tf.divide(numerator, w), c_val)
    return mellow_max


mellow_max_fn = get_mellow_max_fn(250.0)
# (100, 64, 196, 2, 2)
input_tensor = tf.constant([[
    [[[1.2, 2.0], [3.0, 4.0]],
      [[5.0, 6.0], [7.0, 8.0]], 
      [[9.0, 10.0], [11.0, 12.0]]],
      
    [[[13.0, 14.0], [15.0, 16.0]], 
     [[17.0, 18.0], [19.0, 20.0]], 
     [[21.0, 22.0], [23.0, 24.0]]],

    [[[25.0, 26.0], [27.0, 28.0]], 
     [[29.0, 30.0], [31.0, 32.0]], 
     [[33.0, 34.0], [35.0, 36.0]]],

    [[[37.0, 38.0], [39.0, 40.0]], 
     [[41.0, 42.0], [43.0, 44.0]], 
     [[45.0, 46.0], [47.0, 48.0]]]
],
[
    [[[1.2, 2.0], [3.0, 4.0]],
      [[5.0, 6.0], [7.0, 8.0]], 
      [[9.0, 10.0], [11.0, 12.0]]],
      
    [[[13.0, 14.0], [15.0, 16.0]], 
     [[17.0, 18.0], [19.0, 20.0]], 
     [[21.0, 22.0], [23.0, 24.0]]],

    [[[25.0, 26.0], [27.0, 28.0]], 
     [[29.0, 30.0], [31.0, 32.0]], 
     [[33.0, 34.0], [35.0, 36.0]]],

    [[[37.0, 38.0], [39.0, 40.0]], 
     [[41.0, 42.0], [43.0, 44.0]], 
     [[45.0, 46.0], [47.0, 48.0]]]
],
[
    [[[1.2, 2.0], [3.0, 4.0]],
      [[5.0, 6.0], [7.0, 8.0]], 
      [[9.0, 10.0], [11.0, 12.0]]],
      
    [[[13.0, 14.0], [15.0, 16.0]], 
     [[17.0, 18.0], [19.0, 20.0]], 
     [[21.0, 22.0], [23.0, 24.0]]],

    [[[25.0, 26.0], [27.0, 28.0]], 
     [[29.0, 30.0], [31.0, 32.0]], 
     [[33.0, 34.0], [35.0, 36.0]]],

    [[[37.0, 38.0], [39.0, 40.0]], 
     [[41.0, 42.0], [43.0, 44.0]], 
     [[45.0, 46.0], [47.0, 48.0]]]
]], dtype=tf.float32)


print(input_tensor.shape)
input_tensor = tf.reshape(input_tensor, shape=(36,-1))
print(input_tensor.shape)
# Apply the custom_fn using tf.map_fn
output_tensor = tf.vectorized_map(fn=mellow_max_fn, elems=input_tensor, fallback_to_while_loop=False)
# Print the shape of the output tensor
print(output_tensor.shape)  # Output: (4, 3, 3, 1)
print('------------')
output_tensor = tf.reshape(output_tensor, shape=(3,4,3,1))
print(output_tensor)