import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
mse = tf.keras.losses.MeanSquaredError(reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def distance(real, pred):
    if real is None or pred is None:
        return 0
    else:
        return mse(real, pred)


def accuracy_function(real, pred):
    p = tf.argmax(pred, axis=2, output_type=tf.dtypes.int32)  # 这里遇到一个数据类型不匹配的问题，默认output_type=tf.dtypes.int64
    accuracies = tf.equal(real, p)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
