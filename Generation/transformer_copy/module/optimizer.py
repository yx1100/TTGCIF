import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, shrink_factor=10, warmup_steps=4000):  # 4000
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.shrink_factor = shrink_factor

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) / self.shrink_factor

        # return tf.math.rsq
        # rt(self.d_model) * tf.math.minimum(arg1, arg2)
        # return tf.math.rsqrt(1024.0) * tf.math.minimum(arg1, arg2) / 10
