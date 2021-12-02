import time

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from numpy.random import normal

flag = 1
seed = normal(loc=0.0, scale=1.5, size=7)
PCA_model = PCA(n_components=2)


def get_PCA(path):
    array = np.load(path)
    array = array[:4000, :, :]
    print(array.shape)
    _, _, dim = array.shape
    array = np.reshape(array, newshape=[-1, dim])
    reducted_array = PCA_model.fit_transform(array)
    return reducted_array


def getOriginalData(path):
    array = np.load(path)
    array = array[:4000, :, :]
    print(array.shape)
    _, _, dim = array.shape
    array = np.reshape(array, newshape=[-1, dim])
    return array


source_domains = [
    # 'Arts',
    'CD',
    'Digital',
    'Electronic',
    'kindle',
    'Movie',
    'Toy',
    'Video'
]
source1 = source_domains[1]
target_domain = 'Arts'

print(time.strftime('%H:%M:%S', time.localtime(time.time())))

# update
S1 = get_PCA("./data/" + source1 + ".npy") + seed[0]
S1_MMD = get_PCA("./model/" + target_domain + "/" + source1 + "/" + source1 + "_MMD.npy")
S1_Tar_MMD = getOriginalData(
    "./model/" + target_domain + "/" + source1 + "/" + source1 + "_" + target_domain + "_MMD.npy")
S1_Tar_MMD = PCA_model.fit_transform(S1_Tar_MMD)
print("S1 0K.....")

plt.subplot(1, 3, 1)
plt.scatter(S1[:, 1], S1[:, 0], s=1, c='#B8860B', label=source1, alpha=0.3)
plt.legend(loc=0)

plt.subplot(1, 3, 2)
plt.scatter(S1_MMD[:, 1], S1_MMD[:, 0], s=1, c='green', label=source1, alpha=0.3)
plt.legend(loc=0)

plt.subplot(1, 3, 3)
plt.scatter(S1_Tar_MMD[:, 1], S1_Tar_MMD[:, 0], s=1, c='#B8860B', label=source1, alpha=0.3)
plt.legend(loc=0)

print(time.strftime('%H:%M:%S', time.localtime(time.time())))

plt.show()

print(time.strftime('%H:%M:%S', time.localtime(time.time())))
