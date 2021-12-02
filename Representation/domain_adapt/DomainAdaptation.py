# coding=utf-8
import time

import numpy as np
import tensorflow as tf
import DomainUtils as utils

print(time.strftime('%H:%M:%S', time.localtime(time.time())))
domains = {
    1: 'Arts',
    2: 'CDs',
    3: 'Digital',
    4: 'Electronics',
    5: 'Kindle',
    6: 'Movies',
    7: 'Toys',
    8: 'Video'
}

for i in range(1, 9):
    SOURCE_DOMAIN = domains[i]
    TARGET_DOMAIN = domains[4]  # todo 1 更改目标域

    if SOURCE_DOMAIN == TARGET_DOMAIN:
        continue

    TOTAL_STEPS = 2500
    print("Source Domain: %s, Target Domain: %s" % (SOURCE_DOMAIN, TARGET_DOMAIN))

    # 1. 准备input
    source = utils.getDomainInput("../../datasets/6_dataset_embed_review/%s.npy" % SOURCE_DOMAIN)
    target = utils.getDomainInput("../../datasets/6_dataset_embed_review/%s.npy" % TARGET_DOMAIN)
    aligned_source, aligned_target, batch_size, source_mex_len, target_mex_len, dim = utils.alignDomains(source, target)
    print("source shape: ", aligned_source.shape)
    print("target shape: ", aligned_target.shape)
    source_array = np.reshape(aligned_source, newshape=[-1, dim])  # 确保词向量在行方向上
    target_array = np.reshape(aligned_target, newshape=[-1, dim])
    print("source shape: ", source_array.shape)
    print("target shape: ", target_array.shape)
    assert source_array.shape == target_array.shape

    # 2. 构建计算图
    source_holder = tf.placeholder(dtype=tf.float32, shape=[None, dim])
    target_holder = tf.placeholder(dtype=tf.float32, shape=[None, dim])
    source_dense = tf.layers.dense(inputs=source_holder,
                                   units=dim,
                                   activation=tf.nn.sigmoid,
                                   kernel_initializer=tf.variance_scaling_initializer(),
                                   bias_initializer=tf.variance_scaling_initializer(),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    XX, YY, XY, YX = utils.guassian_kernel(source=source_dense, target=target_holder)
    result1 = tf.add(XX, YY)
    result2 = tf.add(XY, YX)
    result3 = tf.subtract(result1, result2)
    loss = tf.reduce_mean(result3)

    my_opt = tf.train.AdagradOptimizer(learning_rate=5)
    train_step = my_opt.minimize(loss)

    # 3. 定义训练图
    sess = tf.Session()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    if utils.isCheckPointExists("./model/" + TARGET_DOMAIN + '/' + SOURCE_DOMAIN + "/", "checkpoint") is not True:
        sess.run(init)
        print("train model...")

        for step in range(TOTAL_STEPS):
            # 2. 存model
            loss_val, _ = sess.run([loss, train_step],
                                   feed_dict={source_holder: source_array, target_holder: target_array})
            print("loss: ", step, loss_val)

        saver.save(sess, "./model/" + TARGET_DOMAIN + '/' + SOURCE_DOMAIN + "/" + "model.ckpt", global_step=step)

        print(time.strftime('%H:%M:%S', time.localtime(time.time())))

    else:
        print("restore model...")
        saver.restore(sess, tf.train.latest_checkpoint('./model/' + TARGET_DOMAIN + '/' + SOURCE_DOMAIN))
        source_data = utils.getDomainInput("../../datasets/6_dataset_embed_review/%s.npy" % SOURCE_DOMAIN)
        batch_size, source_len, dim = source_data.shape
        target_data = utils.getDomainInput("../../datasets/6_dataset_embed_review/%s.npy" % TARGET_DOMAIN)
        _, target_len, _ = target_data.shape

        source_data = np.reshape(source_data, newshape=[-1, dim])
        target_data = np.reshape(target_data, newshape=[-1, dim])

        source_MMD = sess.run(source_dense, feed_dict={source_holder: source_data})
        source_target_MMD = sess.run(source_dense, feed_dict={source_holder: target_data})

        source_MMD = np.reshape(source_MMD, newshape=[batch_size, -1, dim])
        source_target_MMD = np.reshape(source_target_MMD, newshape=[batch_size, -1, dim])

        print("Source_MMD shape: ", source_MMD.shape)
        print("Source_Target_MMD shape: ", source_target_MMD.shape)
        np.save("../../datasets/7_dataset_mmd/" + TARGET_DOMAIN + '/' + SOURCE_DOMAIN + "_MMD.npy", source_MMD)
        np.save("../../datasets/7_dataset_mmd/" + TARGET_DOMAIN + '/' + SOURCE_DOMAIN + "_%s_MMD.npy" % TARGET_DOMAIN,
                source_target_MMD)
    sess.close()
