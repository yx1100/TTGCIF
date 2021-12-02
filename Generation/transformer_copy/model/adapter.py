import tensorflow as tf


class Adapter(tf.keras.layers.Layer):
    def __init__(self, d_model, adapter_size):
        super(Adapter, self).__init__()
        self.adapter_down = tf.keras.layers.Dense(adapter_size, activation='relu')

        self.adapter_up = tf.keras.layers.Dense(d_model)

    def call(self, adapter_input):
        adapted = self.adapter_down(adapter_input)
        adapted = self.adapter_up(adapted)
        adapter_output = tf.add(adapter_input, adapted)

        return adapter_output
