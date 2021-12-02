# coding=utf-8

import matplotlib
import os

matplotlib.use('TkAgg')
import numpy as np
import tensorflow as tf


def guassian_kernel(source, target):
    XY_kernel_sum = XY_type(source, target)
    YX_kernel_sum = YX_type(target, source)
    XX_kernel_sum = XX_type(source)
    YY_kernel_sum = YY_type(target)

    return XX_kernel_sum, YY_kernel_sum, XY_kernel_sum, YX_kernel_sum


def XY_type(x, y):
    l2_distance = tf.square(tf.subtract(x, y))
    kernel_sum = comput_kernel_sum(l2_distance)
    return kernel_sum


def YX_type(y, x):
    l2_distance = tf.square(tf.subtract(x, y))
    kernel_sum = comput_kernel_sum(l2_distance)
    return kernel_sum


def XX_type(x):
    l2_distance = tf.square(tf.subtract(x, x))
    kernel_sum = comput_kernel_sum(l2_distance)
    return kernel_sum


def YY_type(y):
    l2_distance = tf.square(tf.subtract(y, y))
    kernel_sum = comput_kernel_sum(l2_distance)
    return kernel_sum


def comput_kernel_sum(dist, bandwidth_list=[0.25, 0.5, 1, 2, 4]):
    kernel_val = [tf.exp((-1 * dist) / (bandwidth_temp)) for bandwidth_temp in bandwidth_list]
    kernel_sum = tf.reduce_mean(kernel_val, axis=-1)
    return kernel_sum


def getDomainInput(path):
    input = np.load(path)
    return input


def alignDomains(source, target):
    batch_size, source_mex_len, dim = source.shape
    _, target_mex_len, _ = target.shape

    # cut-off策略
    aligned_len = min(source_mex_len, target_mex_len)
    aligned_source = source[:, :aligned_len, :]
    aligned_target = target[:, :aligned_len, :]

    return aligned_source, aligned_target, batch_size, source_mex_len, target_mex_len, dim


def alignShape(common_rows, max_cols, data):
    current_rows, current_cols = data.shape
    if max_cols == current_cols:
        return data
    else:
        add_dim = max_cols - current_cols
        # print add_dim

        average = np.average(data)
        print(average)
        average_pad = np.array([average] * (common_rows * add_dim))
        avargae_pad = np.reshape(average_pad, newshape=[common_rows, add_dim])

        new_data = np.concatenate([data, avargae_pad], axis=1)
        # print new_data.shape
        return new_data


def alignShape_cutOff(common_rows, min_cols, data):
    current_rows, current_cols = data.shape
    if min_cols == current_cols:
        return data
    else:
        sub_dim = current_cols - min_cols
        new_data = data[:common_rows, :-(sub_dim)]
        return new_data


def isCheckPointExists(path, filename):
    return os.path.exists(path + filename)


def testShape(path):
    array = np.load(path)
    print(array.shape)


def shuffle_arr(arr):
    np.random.shuffle(arr)
    return arr


if __name__ == "__main__":
    source = getDomainInput("data_legacy/Arts.npy")
    target = getDomainInput("data_legacy/Toy.npy")

    print("old shape:")
    print("source: ", source.shape)
    print("target: ", target.shape)
    aligned_source, aligned_target, batch_size, source_mex_len, target_mex_len, dim = alignDomains(source, target)

    print("new shape: ")
    print("source: ", aligned_source.shape)
    print("target: ", aligned_source.shape)

    # print "shape check ...."
    #
    # print getDomainInput("./data_legacy/new_Art.npy").shape
    # print getDomainInput("./data_legacy/Digital.npy").shape
    # print getDomainInput("./data_legacy/CD.npy").shape
    # print getDomainInput("./data_legacy/Movie.npy").shape
    # print getDomainInput("./data_legacy/Kindle.npy").shape
    # print getDomainInput("./data_legacy/Video.npy").shape
    # print getDomainInput("./data_legacy/new_E.npy").shape
    # # print getDomainInput("./data_legacy/new_Toy.npy").shape
    # #
    # print "------------------------------------"
    # #
    # testShape("./model/Arts/Art_MMD_subtle.npy")
    # testShape("./model/Digital/Digital_MMD.npy")
    # testShape("./model/CD/CD_MMD.npy")
    # testShape("./model/Movie/Movie_MMD_subtle.npy")
    # testShape("./model/Kindle/Kindle_MMD.npy")
    # testShape("./model/Video/Video_MMD.npy")
    # testShape("./model/Electronic/E_MMD.npy")
    # # testShape("./model/Toy/Toy_MMD_subtle.npy")
