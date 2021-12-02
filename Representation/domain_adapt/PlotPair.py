import time

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from numpy.random import normal

print(time.strftime('%H:%M:%S', time.localtime(time.time())))

flag = 1
seed = normal(loc=0.0, scale=1.5, size=7)
PCA_model = PCA(n_components=2)


def get_PCA(path):
    array = np.load(path)
    print(array.shape)
    _, _, dim = array.shape
    array = np.reshape(array, newshape=[-1, dim])
    reducted_array = PCA_model.fit_transform(array)
    return reducted_array


def getOriginalData(path):
    array = np.load(path)
    print(array.shape)
    _, _, dim = array.shape
    array = np.reshape(array, newshape=[-1, dim])
    return array


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
source1 = domains[1]  # todo 2 更改源域
source2 = domains[2]
source3 = domains[3]
source4 = domains[5]
source5 = domains[6]
source6 = domains[7]
source7 = domains[8]
target_domain = domains[4]  # todo 2 更改目标域

# update
S1 = get_PCA("D:/pycharm/workspace/TPCC/datasets/6_dataset_embed_review/" + source1 + ".npy") + seed[0]
S1_MMD = get_PCA("D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source1 + "_MMD.npy")
S1_Tar_MMD = getOriginalData(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source1 + "_" + target_domain + "_MMD.npy")
print("S1 0K.....")

S2 = get_PCA("D:/pycharm/workspace/TPCC/datasets/6_dataset_embed_review/" + source2 + ".npy") + seed[1]
S2_MMD = get_PCA(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source2 + "_MMD.npy")
S2_Tar_MMD = getOriginalData(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source2 + "_" + target_domain + "_MMD.npy")
print("S2 0K.....")

S3 = get_PCA("D:/pycharm/workspace/TPCC/datasets/6_dataset_embed_review/" + source3 + ".npy") + seed[2]
S3_MMD = get_PCA(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source3 + "_MMD.npy")
S3_Tar_MMD = getOriginalData(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source3 + "_" + target_domain + "_MMD.npy")
print("S3 0K.....")

S4 = get_PCA("D:/pycharm/workspace/TPCC/datasets/6_dataset_embed_review/" + source4 + ".npy") + seed[3]
S4_MMD = get_PCA(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source4 + "_MMD.npy")
S4_Tar_MMD = getOriginalData(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source4 + "_" + target_domain + "_MMD.npy")
print("S4 0K.....")

S5 = get_PCA("D:/pycharm/workspace/TPCC/datasets/6_dataset_embed_review/" + source5 + ".npy") + seed[4]
S5_MMD = get_PCA(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source5 + "_MMD.npy")
S5_Tar_MMD = getOriginalData(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source5 + "_" + target_domain + "_MMD.npy")
print("S5 0K.....")

S6 = get_PCA("D:/pycharm/workspace/TPCC/datasets/6_dataset_embed_review/" + source6 + ".npy") + seed[5]
S6_MMD = get_PCA(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source6 + "_MMD.npy")
S6_Tar_MMD = getOriginalData(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source6 + "_" + target_domain + "_MMD.npy")
print("S6 0K.....")

S7 = get_PCA("D:/pycharm/workspace/TPCC/datasets/6_dataset_embed_review/" + source7 + ".npy") + seed[6]
S7_MMD = get_PCA(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source7 + "_MMD.npy")
S7_Tar_MMD = getOriginalData(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + source7 + "_" + target_domain + "_MMD.npy")
print("S7 0K.....")

Tar = get_PCA("D:/pycharm/workspace/TPCC/datasets/6_dataset_embed_review/" + target_domain + ".npy")

target_MMDs = np.stack([S1_Tar_MMD, S2_Tar_MMD, S3_Tar_MMD, S4_Tar_MMD, S5_Tar_MMD, S6_Tar_MMD, S7_Tar_MMD])
print(target_MMDs.shape)
Tar_MMD = np.average(target_MMDs, axis=0)
print(Tar_MMD.shape)
np.save(
    "D:/pycharm/workspace/TPCC/datasets/7_dataset_mmd/" + target_domain + "/" + target_domain + '_' + target_domain + "_MMD.npy",
    np.reshape(Tar_MMD, newshape=[500, -1, 512]))
Tar_MMD = PCA_model.fit_transform(Tar_MMD)
print(Tar_MMD.shape)

if flag == 1:
    plt.subplot(1, 2, 1)
    plt.scatter(S1[:, 1], S1[:, 0], s=1, c='#B8860B', label=source1, alpha=0.2)
    plt.scatter(S2[:, 1], S2[:, 0], s=1, c='#98FB98', label=source2, alpha=0.2)
    plt.scatter(S3[:, 1], S3[:, 0], s=1, c='#FF4500', label=source3, alpha=0.3)
    plt.scatter(S4[:, 1], S4[:, 0], s=1, c='#9370DB', label=source4, alpha=0.3)
    plt.scatter(S5[:, 1], S5[:, 0], s=1, c='#00FFFF', label=source5, alpha=0.4)
    plt.scatter(S6[:, 1], S6[:, 0], s=1, c='#808080', label=source6, alpha=0.4)
    plt.scatter(S7[:, 1], S7[:, 0], s=1, c='#FF1493', label=source7, alpha=0.45)
    plt.scatter(Tar[:, 1], Tar[:, 0], s=1, c='#0000CD', label=target_domain + "(Target)", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(S1_MMD[:, 1], S1_MMD[:, 0], s=1, c='#B8860B', label=source1 + "->" + target_domain, alpha=0.1)
    plt.scatter(S2_MMD[:, 1], S2_MMD[:, 0], s=1, c='#98FB98', label=source2 + "->" + target_domain, alpha=0.2)
    plt.scatter(S3_MMD[:, 1], S3_MMD[:, 0], s=1, c='#FF4500', label=source3 + "->" + target_domain, alpha=0.3)
    plt.scatter(S4_MMD[:, 1], S4_MMD[:, 0], s=1, c='#9370DB', label=source4 + "->" + target_domain, alpha=0.4)
    plt.scatter(S5_MMD[:, 1], S5_MMD[:, 0], s=1, c='#00FFFF', label=source5 + "->" + target_domain, alpha=0.5)
    plt.scatter(S6_MMD[:, 1], S6_MMD[:, 0], s=1, c='#808080', label=source6 + "->" + target_domain, alpha=0.6)
    plt.scatter(S7_MMD[:, 1], S7_MMD[:, 0], s=1, c='#FF1493', label=source7 + "->" + target_domain, alpha=0.7)
    plt.scatter(Tar_MMD[:, 1], Tar_MMD[:, 0], s=1, c='#0000CD', label=target_domain + "(Target)", alpha=0.8)
    plt.legend()

if flag == 2:
    plt.subplot(2, 3, 1)
    plt.scatter(S1[:, 1], S1[:, 0], s=1, c='#B8860B', label=source1, alpha=0.3)
    plt.scatter(Tar[:, 1], Tar[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.6)
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.scatter(S2[:, 1], S2[:, 0], s=1, c='green', label=source2, alpha=0.3)
    plt.scatter(Tar[:, 1], Tar[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.6)
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.scatter(S3[:, 1], S3[:, 0], s=1, c='#FF4500', label=source3, alpha=0.3)
    plt.scatter(Tar[:, 1], Tar[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.6)
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.scatter(S1_MMD[:, 1], S1_MMD[:, 0], s=1, c='#B8860B', label=source1 + '->' + target_domain, alpha=0.3)
    plt.scatter(Tar_MMD[:, 1], Tar_MMD[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.6)
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.scatter(S2_MMD[:, 1], S2_MMD[:, 0], s=1, c='green', label=source2 + '->' + target_domain, alpha=0.3)
    plt.scatter(Tar_MMD[:, 1], Tar_MMD[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.6)
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.scatter(S3_MMD[:, 1], S3_MMD[:, 0], s=1, c='#FF4500', label=source3 + '->' + target_domain, alpha=0.3)
    plt.scatter(Tar_MMD[:, 1], Tar_MMD[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.6)
    plt.legend()

if flag == 3:
    plt.subplot(2, 4, 1)
    plt.scatter(S4[:, 1], S4[:, 0], s=1, c='#9370DB', label=source4, alpha=0.3)
    plt.scatter(Tar[:, 1], Tar[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.5)
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.scatter(S5[:, 1], S5[:, 0], s=1, c='#00FFFF', label=source5, alpha=0.3)
    plt.scatter(Tar[:, 1], Tar[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.5)
    plt.legend()

    plt.subplot(2, 4, 3)
    plt.scatter(S6[:, 1], S6[:, 0], s=1, c='#808080', label=source6, alpha=0.3)
    plt.scatter(Tar[:, 1], Tar[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.5)
    plt.legend()

    plt.subplot(2, 4, 4)
    plt.scatter(S7[:, 1], S7[:, 0], s=1, c='#FF1493', label=source7, alpha=0.3)
    plt.scatter(Tar[:, 1], Tar[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.5)
    plt.legend()

    plt.subplot(2, 4, 5)
    plt.scatter(S4_MMD[:, 1], S4_MMD[:, 0], s=1, c='#9370DB', label=source4 + '->' + target_domain, alpha=0.3)
    plt.scatter(Tar_MMD[:, 1], Tar_MMD[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.5)
    plt.legend()

    plt.subplot(2, 4, 6)
    plt.scatter(S5_MMD[:, 1], S5_MMD[:, 0], s=1, c='#00FFFF', label=source5 + '->' + target_domain, alpha=0.3)
    plt.scatter(Tar_MMD[:, 1], Tar_MMD[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.5)
    plt.legend()

    plt.subplot(2, 4, 7)
    plt.scatter(S6_MMD[:, 1], S6_MMD[:, 0], s=1, c='#808080', label=source6 + '->' + target_domain, alpha=0.3)
    plt.scatter(Tar_MMD[:, 1], Tar_MMD[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.5)
    plt.legend()

    plt.subplot(2, 4, 8)
    plt.scatter(S7_MMD[:, 1], S7_MMD[:, 0], s=1, c='#FF1493', label=source7 + '->' + target_domain, alpha=0.3)
    plt.scatter(Tar_MMD[:, 1], Tar_MMD[:, 0], s=1, c='#0000CD', label=target_domain, alpha=0.5)
    plt.legend()

plt.show()
