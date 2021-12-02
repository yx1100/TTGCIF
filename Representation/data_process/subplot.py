# 导入相关库
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def get_numpy(path):
    with open(path, 'rb') as f:
        n = pickle.load(f)
    n = n.numpy()
    n = n.transpose()
    print('numpy shape:', n.shape)

    x_dr = pca.fit_transform(n)
    print('降维shape: ', x_dr.shape)

    return x_dr


def get_mmd_numpy(path):
    n = np.load(path)
    x_dr = pca.fit_transform(n)
    print('MMD降维shape:', x_dr.shape)
    return x_dr


def draw_2d():
    # 2D
    plt.figure(figsize=(6.4, 9.6))

    # plt.subplot(2, 1, 1)
    # plt.scatter(x_dr1[:, 1], x_dr1[:, 0], s=1, c='#B8860B', label='Arts')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.scatter(x_dr11[:, 1], x_dr11[:, 0], s=1, c='#B8860B', label='Arts->Toy')
    # plt.legend()
    # plt.subplot(2, 1, 1)
    # plt.scatter(x_dr2[:, 1], x_dr2[:, 0], s=1, c='#98FB98', label='CD')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.scatter(x_dr22[:, 1], x_dr22[:, 0], s=1, c='#98FB98', label='CD->Toy')
    # plt.legend()

    # plt.subplot(2, 1, 1)
    # plt.scatter(x_dr3[:, 1], x_dr3[:, 0], s=1, c='#FF4500', label='Digital')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.scatter(x_dr33[:, 1], x_dr33[:, 0], s=1, c='#FF4500', label='Digital->Toy')
    # plt.legend()
    # plt.subplot(2, 1, 1)
    # plt.scatter(x_dr4[:, 1], x_dr4[:, 0], s=1, c='#9370DB', label='Ele')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.scatter(x_dr44[:, 1], x_dr44[:, 0], s=1, c='#9370DB', label='Ele->Toy')
    # plt.legend()

    # plt.subplot(2, 1, 1)
    # plt.scatter(x_dr5[:, 1], x_dr5[:, 0], s=1, c='#00FFFF', label='Kindle')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.scatter(x_dr55[:, 1], x_dr55[:, 0], s=1, c='#00FFFF', label='Kindle->Toy')
    # plt.legend()
    # plt.subplot(2, 1, 1)
    # plt.scatter(x_dr6[:, 1], x_dr6[:, 0], s=1, c='#808080', label='Movie')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.scatter(x_dr66[:, 1], x_dr66[:, 0], s=1, c='#808080', label='Movie->Toy')
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.scatter(x_dr8[:, 1], x_dr8[:, 0], s=1, c='#FF1493', label='Video')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.scatter(x_dr88[:, 1], x_dr88[:, 0], s=1, c='#FF1493', label='Video->Toy')
    # plt.legend()
    plt.subplot(2, 1, 1)
    plt.scatter(x_dr7[:, 1], x_dr7[:, 0], s=1, c='#0000CD', label='Toy (Target Domain)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.scatter(x_dr77[:, 1], x_dr77[:, 0], s=1, c='#0000CD', label='Toy (Target Domain)')
    plt.legend()

    plt.show()


# 调用PCA
pca = PCA(n_components=2, whiten=True)
# pca = PCA(n_components=3, whiten=True)

# art
x_dr1 = get_numpy('../数据集/处理后的数据集/Arts_Crafts_and_Sewing/reviews/matrix.pickle')
x_dr11 = get_mmd_numpy('数据集/MMD/Art_MMD_subtle.npy')

# cd
x_dr2 = get_numpy('../数据集/处理后的数据集/CDs_and_Vinyl/reviews/matrix.pickle')
x_dr22 = get_mmd_numpy('数据集/MMD/CD_MMD_new.npy')

# digital
x_dr3 = get_numpy('../数据集/处理后的数据集/Digital_Music/reviews/matrix.pickle')
x_dr33 = get_mmd_numpy('数据集/MMD/Digital_MMD_new.npy')

# Ele
x_dr4 = get_numpy('../数据集/处理后的数据集/Electronics/reviews/matrix.pickle')
x_dr44 = get_mmd_numpy('数据集/MMD/E_MMD_new.npy')

# kindle
x_dr5 = get_numpy('../数据集/处理后的数据集/Kindle_Store/reviews/matrix.pickle')
x_dr55 = get_mmd_numpy('数据集/MMD/Kindle_MMD_new.npy')

# movie
x_dr6 = get_numpy('../数据集/处理后的数据集/Movies_and_TV/reviews/matrix.pickle')
x_dr66 = get_mmd_numpy('数据集/MMD/Movie_MMD_subtle.npy')

# toy
x_dr7 = get_numpy('../数据集/处理后的数据集/Toys_and_Games/reviews/matrix.pickle')
x_dr77 = get_mmd_numpy('数据集/MMD/Toy_MMD_subtle.npy')

# video
x_dr8 = get_numpy('../数据集/处理后的数据集/Video_Games/reviews/matrix.pickle')
x_dr88 = get_mmd_numpy('数据集/MMD/Video_MMD_new.npy')

draw_2d()
