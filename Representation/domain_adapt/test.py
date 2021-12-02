import numpy as np

# n = np.load('D:\pycharm\workspace\process_mmd\Domain_Adapt\model\Arts\Video\Video_Art_MMD.npy')
# print(n.shape)

a = np.arange(1, 9).reshape((2, 4))
b = np.arange(1, 9).reshape((2, 4))
c = np.arange(1, 9).reshape((2, 4))

s = np.stack((a, b, c), axis=0)
print(s, '\n')

ss = np.reshape(s, newshape=(-1, 4))
print(ss, '\n')
#
# sss = np.average(ss, axis=0)
# print(sss, '\n')
#
# print(ss.shape)
#
# print(sss)
