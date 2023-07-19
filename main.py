import tensorflow as tf
import math
import sys

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_full[:5000]/255.0, x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test/255.0

tensorboard_writer = tf.summary.create_file_writer('tensorboard/')

class ValidationLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with tensorboard_writer.as_default():
            val_loss = logs['val_loss']
            tf.summary.scalar('val_loss', val_loss, step=epoch)
            tensorboard_writer.flush()

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, custom_layer):
        super(CustomCallback, self).__init__()
        self.custom_layer = custom_layer

    def on_train_batch_end(self, batch, logs=None):
        custom_variable_value = self.custom_layer.w.numpy()
        with tensorboard_writer.as_default():
            tf.summary.scalar('w_value', data=custom_variable_value, step=batch)        # You can also log the value to other logging frameworks (e.g., TensorBoard)
            tensorboard_writer.flush()
    # Alternatively, you can use on_epoch_end to log the variable at the end of each epoch:
    # def on_epoch_end(self, epoch, logs=None):
    #     custom_variable_value = self.custom_layer.custom_variable.numpy()
    #     tf.summary.scalar('custom_variable', data=custom_variable_value, step=epoch)
    #     # You can also log the value to other logging frameworks (e.g., TensorBoard)


loss = tf.keras.losses.SparseCategoricalCrossentropy()
opt = tf.keras.optimizers.Adam()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


class CustomMaxPooling2D(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomMaxPooling2D, self).__init__()
        self.padding_format = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        self.transpose_format = tf.constant([0, 3, 1, 2])
        self.reverse_transpose_format = tf.constant([0, 2, 3, 1])
    def build(self, input_shape):
        self.batch_size, self.height, self.width, self.channels = input_shape
        self.output_width = math.floor(((self.height-1)/2)+1)     
    def call(self, inputs):
        if self.width%self.output_width!=0:
            inputs = tf.pad(tensor=inputs, paddings=self.padding_format, mode='constant')
        inputs = tf.transpose(inputs, perm=self.transpose_format)
        reshaped_tensor = tf.reshape(inputs, shape=(self.batch_size, self.channels, int(((2*self.output_width)**2)/4), 2, 2))
        max_pooled_tensor = tf.reduce_max(reshaped_tensor, axis=(3, 4), keepdims=True)
        output_tensor = tf.reshape(max_pooled_tensor, shape=(self.batch_size, self.channels, self.output_width, self.output_width))
        return tf.transpose(output_tensor, perm=self.reverse_transpose_format)
    



class GreedyMaxPooling2D(tf.keras.layers.Layer):
    def __init__(self):
        super(GreedyMaxPooling2D, self).__init__()
        self.padding_format = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        self.transpose_format = tf.constant([0, 3, 1, 2])
        self.reverse_transpose_format = tf.constant([0, 2, 3, 1])
    def build(self, input_shape):
        self.batch_size, self.height, self.width, self.channels = input_shape
        self.output_width = math.floor(((self.height-1)/2)+1)  
        self.epsilon = tf.Variable(initial_value=0.0, trainable=True, name="eps")   
    def call(self, inputs):
        if self.width%self.output_width!=0:
            inputs = tf.pad(tensor=inputs, paddings=self.padding_format, mode='constant')
        inputs = tf.transpose(inputs, perm=self.transpose_format)
        reshaped_tensor = tf.reshape(inputs, shape=(self.batch_size, self.channels, int(((2*self.output_width)**2)/4), 2, 2))
        max_pooled_tensor = tf.reduce_max(reshaped_tensor, axis=(3, 4), keepdims=True)
        output_tensor = tf.reshape(max_pooled_tensor, shape=(self.batch_size, self.channels, self.output_width, self.output_width))
        output_tensor = tf.add(output_tensor, self.epsilon)
        return tf.transpose(output_tensor, perm=self.reverse_transpose_format)

























def get_mellow_max_fn(base_value):
    @tf.function
    def mellow_max(x, tuning_param):
        w = tf.multiply(base_value, tuning_param)
        c_val = tf.reduce_max(x)
        numerator = tf.math.log(tf.multiply(0.25, tf.reduce_sum(tf.math.exp([tf.multiply(w, x[0]-c_val), tf.multiply(w, x[1]-c_val), tf.multiply(w, x[2]-c_val), tf.multiply(w, x[3]-c_val)]))))
        return tf.add(tf.divide(numerator, w), c_val)
    return mellow_max



class TrainableMellowMaxPooling2D(tf.keras.layers.Layer):
    def __init__(self, base_val):
        super(TrainableMellowMaxPooling2D, self).__init__()
        self.padding_format = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        self.transpose_format = tf.constant([0, 3, 1, 2])
        self.reverse_transpose_format = tf.constant([0, 2, 3, 1])
        self.base_val = base_val
        #self.mellow_max_fn = get_mellow_max_fn(base_value)
    def build(self, input_shape):
        self.batch_size, self.height, self.width, self.channels = input_shape
        self.output_width = math.floor(((self.height-1)/2)+1)     
        self.w = tf.Variable(initial_value=1.0, trainable=True, name="w")
        #self.tuning_param = tf.Variable(name="mellow_tuning_param", initial_value=1.0, trainable=True)
    def call(self, inputs):
        if self.width%self.output_width!=0:
            inputs = tf.pad(tensor=inputs, paddings=self.padding_format, mode='constant')
        inputs = tf.transpose(inputs, perm=self.transpose_format)
        reshaped_tensor = tf.reshape(inputs, shape=(self.batch_size * self.channels * int(((2*self.output_width)**2)/4), -1))
        mellow_max_pooled_tensor = tf.vectorized_map(fn=self.mellow_max, elems=reshaped_tensor, fallback_to_while_loop=False)
        output_tensor = tf.reshape(mellow_max_pooled_tensor, shape=(self.batch_size, self.channels, self.output_width, self.output_width))
        return tf.transpose(output_tensor, perm=self.reverse_transpose_format)
    def mellow_max(self, x):
        c_val = tf.reduce_max(x)
        numerator = tf.math.log(tf.multiply(0.25, tf.reduce_sum(tf.math.exp([tf.multiply(tf.multiply(self.w, self.base_val), x[0]-c_val), tf.multiply(tf.multiply(self.w, self.base_val), x[1]-c_val), tf.multiply(tf.multiply(self.w, self.base_val), x[2]-c_val), tf.multiply(tf.multiply(self.w, self.base_val), x[3]-c_val)]))))
        return tf.add(tf.divide(numerator, tf.multiply(self.w, self.base_val)), c_val)


'''
class CustomFixedMellowMaxPooling2D(tf.keras.layers.Layer):
    def __init__(self, base_value):
        super(CustomFixedMellowMaxPooling2D, self).__init__()
        self.padding_format = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        self.transpose_format = tf.constant([0, 3, 1, 2])
        self.reverse_transpose_format = tf.constant([0, 2, 3, 1])
        self.mellow_max_fn = get_mellow_max_fn(base_value)
    def build(self, input_shape):
        self.batch_size, self.height, self.width, self.channels = input_shape
        self.output_width = math.floor(((self.height-1)/2)+1)     
    def call(self, inputs):
        if self.width%self.output_width!=0:
            inputs = tf.pad(tensor=inputs, paddings=self.padding_format, mode='constant')
        inputs = tf.transpose(inputs, perm=self.transpose_format)
        reshaped_tensor = tf.reshape(inputs, shape=(self.batch_size * self.channels * int(((2*self.output_width)**2)/4), -1))
        mellow_max_pooled_tensor = tf.vectorized_map(fn=self.mellow_max_fn, elems=reshaped_tensor, fallback_to_while_loop=False)
        output_tensor = tf.reshape(mellow_max_pooled_tensor, shape=(self.batch_size, self.channels, self.output_width, self.output_width))
        return tf.transpose(output_tensor, perm=self.reverse_transpose_format)'''
    


mellow_max_pooling_layer_1 = TrainableMellowMaxPooling2D(1000.0)   
mellow_max_pooling_layer_2 = TrainableMellowMaxPooling2D(1000.0)
mellow_max_pooling_layer_3 = TrainableMellowMaxPooling2D(1000.0)

custom_callback = CustomCallback(mellow_max_pooling_layer_1)


model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28,28,1), batch_size=100),
    tf.keras.layers.Conv2D(64,7,activation="relu",padding="same",input_shape=[28,28,1]),
    GreedyMaxPooling2D(),
    tf.keras.layers.Conv2D(128,3,activation="relu",padding="same"),
    tf.keras.layers.Conv2D(128,3,activation="relu",padding="same"),
    GreedyMaxPooling2D(),
    tf.keras.layers.Conv2D(256,3,activation="relu",padding="same"),
    tf.keras.layers.Conv2D(256,3,activation="relu",padding="same"),
    GreedyMaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer=opt, loss=loss)

'''

print('------------------------------\n')
print('INITIAL W VALUES:\n')
print('mellow_max_pooling_layer_1: '+str(mellow_max_pooling_layer_1.variables)+'\n')
print('mellow_max_pooling_layer_2: '+str(mellow_max_pooling_layer_2.variables)+'\n')
print('mellow_max_pooling_layer_3: '+str(mellow_max_pooling_layer_3.variables)+'\n')
print('------------------------------\n')
'''

model.fit(x_train, y_train, batch_size=100, epochs=10000,shuffle=True, validation_data=(x_valid, y_valid), callbacks=[early_stopping, ValidationLossCallback()])

'''print('------------------------------\n')
print('FINAL W VALUES:\n')
print('mellow_max_pooling_layer_1: '+str(mellow_max_pooling_layer_1.w)+'\n')
print('mellow_max_pooling_layer_2: '+str(mellow_max_pooling_layer_2.w)+'\n')
print('mellow_max_pooling_layer_3: '+str(mellow_max_pooling_layer_3.w)+'\n')
print('------------------------------\n\n')'''


predictions = model.predict(x_test, batch_size=100)

# Step 5: Evaluate the predictions
accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(tf.argmax(predictions, axis=1), y_test)
accuracy_result = accuracy.result().numpy()

print("Accuracy: {:.2f}%".format(accuracy_result * 100))

model.save_weights('models/mellowmaxpoolmodel.h5')